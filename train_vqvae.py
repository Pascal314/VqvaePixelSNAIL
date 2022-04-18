from vqvae import ResidualStack, Encoder, Decoder, VQVAEModel
from typing import Generator, Mapping, Tuple, NamedTuple, Sequence
import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import dataset
import sys
import tensorflow_datasets as tfds
import tensorflow as tf
import pickle
tf.config.set_visible_devices([], 'GPU')
Batch = Mapping[str, np.ndarray]

# Set hyper-parameters.
batch_size = 32
image_size = 28

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
num_training_updates = 100000

num_hiddens = 16
num_residual_hiddens = 8
num_residual_layers = 2

# This value is not that important, usually 64 works.
# This will not change the capacity in the information-bottleneck.
embedding_dim = 4 

# The higher this value, the higher the capacity in the information bottleneck.
num_embeddings = 4

# commitment_cost should be set appropriately. It's often useful to try a couple
# of values. It mostly depends on the scale of the reconstruction cost
# (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.
commitment_cost = 2.5 # was 2.5

# Use EMA updates for the codebook (instead of the Adam optimizer).
# This typically converges faster, and makes the model less dependent on choice
# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
vq_use_ema = True

# This is only used for EMA updates.
decay = 0.99

learning_rate = 3e-4
learning_rate = optax.linear_schedule(1e-3, 3e-5, 100000)
random_seed = 42

def preprocess(data_dict):
    data_dict['image'] = tf.cast(data_dict['image'], dtype=tf.float32) / 255 - 0.5
    return data_dict

def save_params(params, state, name):
    with open(name, 'wb') as f:
        pickle.dump((params, state), f)

if __name__ == "__main__":
    # # Data Loading.
    ds = tfds.load("mnist", split=tfds.Split.TRAIN, shuffle_files=False,
                   read_config=tfds.ReadConfig(shuffle_seed=random_seed))
    ds = ds.map(preprocess).shuffle(1000).repeat().batch(batch_size)
    ds = iter(tfds.as_numpy(ds))
    train_dataset = ds
    del ds
    train_data_variance = np.var(next(train_dataset)['image'][0])

    # # Build modules.
    def f():
        encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
        decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)
        pre_vq_conv1 = hk.Conv2D(
            output_channels=embedding_dim,
            kernel_shape=(1, 1),
            stride=(1, 1),
            name="to_vq")

        if vq_use_ema:
            vq_vae = hk.nets.VectorQuantizerEMA(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                decay=decay)
        else:
            vq_vae = hk.nets.VectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost)


        model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                            data_variance=train_data_variance)
        return model.multitransform()

    fwd = hk.multi_transform_with_state(f)
    forward, encode, decode = fwd.apply
    optimizer = optax.adam(learning_rate)

    def categorical_KL(encodings):
        p = jnp.clip(jnp.sum(encodings.reshape(batch_size, -1, 4), axis=1) / 49, 1e-12, 1-1e-12)
        return jnp.sum(p * jnp.log(p) - p * jnp.log(0.2))

    @jax.jit
    def train_step(params, state, opt_state, data):
        def adapt_forward(params, state, data):
            # Pack model output and state together.
            data = data['image']
            model_output, state = forward(params, state, None, data, is_training=True)
            kl_loss = categorical_KL(model_output['vq_output']['encodings'])
            loss = model_output['loss'] + kl_loss
            model_output['kl_loss'] = kl_loss
            return loss, (model_output, state)

        grads, (model_output, state) = (
            jax.grad(adapt_forward, has_aux=True)(params, state, data))

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, state, opt_state, model_output

    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []
    kl_loss = []

    rng = jax.random.PRNGKey(42)
    params, state = fwd.init(rng, next(train_dataset)['image'], is_training=True)
    opt_state = optimizer.init(params)

    model_name = 'models/trained_vqvae.pkl'
    train = False
    if train:
        try:
            for step in range(1, num_training_updates + 1):
                data = next(train_dataset)
                params, state, opt_state, train_results = (
                train_step(params, state, opt_state, data))

                # train_results = jax.device_get(train_results)
                train_losses.append(train_results['loss'])
                train_recon_errors.append(train_results['recon_error'])
                train_perplexities.append(train_results['vq_output']['perplexity'])
                train_vqvae_loss.append(train_results['vq_output']['loss'])
                kl_loss.append(train_results['kl_loss'])

                if step % 100 == 0:
                    print(f'[Step {step}/{num_training_updates}] ' + 
                        ('train loss: %f ' % np.mean(train_losses[-100:])) +
                        ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
                        ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
                        ('vqvae loss: %.3f ' % np.mean(train_vqvae_loss[-100:])) +
                        ('kl_loss: %.3f' % np.mean(kl_loss[-100:])))
        except KeyboardInterrupt:
            print('Stopped at iteration', step)
        save_params(params, state, model_name)
    else: 
        with open(model_name, 'rb') as f:
            (params, state) = pickle.load(f)
        print('Params loaded')

    n_test = 20
    # test_ds = tfds.load("mnist", split=tfds.Split.TEST, shuffle_files=False,
    #             read_config=tfds.ReadConfig(shuffle_seed=random_seed))
    # test_ds = test_ds.map(preprocess).shuffle(1000).repeat().batch(batch_size)
    # test_ds = iter(tfds.as_numpy(test_ds))
    # test_batch = next(test_ds)['image']


    x = next(train_dataset)['image']
    model_output, state = forward(params, state, None, x, is_training=False)
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_test)
    fig_size = 1
    fig.set_size_inches(n_test * fig_size, n_rows * fig_size)

    y = (model_output['x_recon'])

    z = model_output['vq_output']['encoding_indices']

    samples = np.random.randint(0, num_embeddings, size=(20, 7, 7))
    decoded, state = decode(params, state, None, samples)

    # print(decoded[0])

    for i in range(n_test):
        axes[0, i].matshow(x[i])
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)
        axes[1, i].matshow(z[i])
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
        axes[2, i].matshow(np.clip(y[i], -0.5, 0.5))
        axes[2, i].get_xaxis().set_visible(False)
        axes[2, i].get_yaxis().set_visible(False)
        axes[3, i].matshow(samples[i])
        axes[3, i].get_xaxis().set_visible(False)
        axes[3, i].get_yaxis().set_visible(False)
        axes[4, i].matshow(np.clip(decoded[i].squeeze(), -0.5, 0.5))
        axes[4, i].get_xaxis().set_visible(False)
        axes[4, i].get_yaxis().set_visible(False)
    plt.show()

    def encode_and_save_ds(ds, save_name):
        images= []
        labels= []

        for batch in ds:
            encoded, _ = encode(params, state, None, batch['image'])
            images.append(encoded)
            labels.append(batch['label'])
        
        dataset = {'image': np.concatenate(images, axis=0), 'label': np.concatenate(labels, axis=0)}
        with open(save_name, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    splits = {
        'train': tfds.Split.TRAIN,
        'test': tfds.Split.TEST
    }

    for split_name, split_enum in splits.items():
        ds = tfds.load("mnist", split=split_enum, shuffle_files=False,
                    read_config=tfds.ReadConfig(shuffle_seed=random_seed))
        ds = ds.map(preprocess).batch(128)
        ds = iter(tfds.as_numpy(ds))
        encode_and_save_ds(ds, 'vqvae_encoded_mnist_'+split_name+'.pkl')
        print(split_name, 'is done')
