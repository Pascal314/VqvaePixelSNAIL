import sys
from PixelSNAIL import PixelSNAIL, make_sample_func
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
import tqdm
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
tf.config.set_visible_devices([], 'GPU')

from vqvae import ResidualStack, Encoder, Decoder, VQVAEModel
from typing import Generator, Mapping, Tuple, NamedTuple, Sequence
import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

Batch = Mapping[str, np.ndarray]

# Set hyper-parameters.
batch_size = 32
image_size = 28

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
num_training_updates = 10000
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
batch_size = 64
random_seed = 42

train = True
snail_name = 'vqvaesnail/vqvae_snail.pkl'
shape = (7, 7)


def load_vqvae_decoder():
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
                            data_variance=1)
        return model.multitransform()

    fwd = hk.multi_transform_with_state(f)
    forward, encode, decode = fwd.apply
    with open('vqvaesnail/trained_vqvae.pkl', 'rb') as f:
        params, state = pickle.load(f)

    decoder = lambda x: decode(params, state, None, x.astype(np.int32))[0]
    return decoder

def show_image_grid(images):
    n, m = images.shape[0:2]
    fig, axes = plt.subplots(n, m)
    for i in range(n):
        for j in range(m):
            axes[i, j].imshow(images[i, j].squeeze())
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
    return fig, axes

def save_params(params, name):
    with open(name, 'wb') as f:
        pickle.dump(params, f)

def load_params(name):
    with open(name, 'rb') as f:
        params = pickle.load(f)
    return params

def load_data(split):
    with open(f'vqvaesnail/vqvae_encoded_mnist_{split}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    ds = tf.data.Dataset.from_tensor_slices(load_data('train'))
    ds = ds.shuffle(1000).repeat().batch(batch_size)
    ds = iter(tfds.as_numpy(ds))

    val_ds = tf.data.Dataset.from_tensor_slices(load_data('test'))
    val_ds = val_ds.shuffle(1000).repeat().batch(batch_size)
    val_ds = iter(tfds.as_numpy(val_ds))

    def forward(x):
        x = jax.nn.one_hot(x, num_embeddings)
        y = PixelSNAIL(num_embeddings, M_blocks=8, R_repeats=0, D_filters=64, key_size=4, value_size=32)(x)
        return y

    batch = next(ds)
    x = batch['image']
    print(x.shape)

    pixelsnail = hk.without_apply_rng(hk.transform(forward))
    params = pixelsnail.init(jax.random.PRNGKey(28), x)
    output = pixelsnail.apply(params, x)

    if train:
        learning_rate = optax.linear_schedule(1e-3, 5e-5, 10000)

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        def loss_fn(params, batch):
            x = batch['image']
            output = pixelsnail.apply(params, x)
            likelihood = jnp.mean(jnp.sum(optax.softmax_cross_entropy(output, jax.nn.one_hot(x, num_embeddings)), axis=(1, 2)), axis=0)
            return likelihood

        @jax.jit
        def update(params, opt_state, batch):
            likelihood, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return likelihood, params, opt_state

        losses = []
        valid_loss = []
        pbar = tqdm.trange(10000 + 1)
        try:
            for i in pbar:
                batch = next(ds)
                loss, params, opt_state = update(params, opt_state, batch)
                losses.append(loss)

                if i % 10 == 0:
                    val_batch = next(val_ds)
                    valid_loss.append(loss_fn(params, val_batch))
                    pbar.set_postfix_str(s=f'loss: {np.mean(losses[-30:]):2f}, val_loss: {np.mean(valid_loss[-3:]):2f}')
        except KeyboardInterrupt:
            print("Stopped at iteration", i)


        save_params(params, name=snail_name)
    else:
        params = load_params(snail_name)

    output = pixelsnail.apply(params, batch['image'])
    show_image_grid(np.concatenate([np.argmax(output[:32], -1), batch['image'][:32]], axis=0).reshape(8, 8, *shape))

    generate_samples = make_sample_func(pixelsnail, params, x, 64)
    samples = generate_samples(jax.random.PRNGKey(42))
    show_image_grid(samples.reshape(8, 8, *shape))

    decoder = load_vqvae_decoder()
    decoded = np.clip(decoder(samples), -0.5, 0.5)
    show_image_grid(decoded.reshape(8, 8, *decoded.shape[1:]))

    plt.show()