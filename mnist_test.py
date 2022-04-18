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

batch_size = 64
random_seed = 42
train = True
load = True
save_name = 'models/model_small'
shape = (28, 28, 1)

def preprocess_and_binarize(data_dict):
    data_dict['image'] = tf.cast(data_dict['image'] > 128, tf.float32)[::1, ::1]
    return data_dict

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

if __name__ == "__main__":
    ds = tfds.load("mnist", split=tfds.Split.TRAIN, shuffle_files=False,
                read_config=tfds.ReadConfig(shuffle_seed=random_seed))
    ds = ds.map(preprocess_and_binarize).shuffle(1000).repeat().batch(batch_size)
    ds = iter(tfds.as_numpy(ds))

    val_ds = tfds.load("mnist", split=tfds.Split.TEST, shuffle_files=False,
            read_config=tfds.ReadConfig(shuffle_seed=random_seed))
    val_ds = val_ds.map(preprocess_and_binarize).shuffle(1000).repeat().batch(256)
    val_ds = iter(tfds.as_numpy(val_ds))

    def forward(x):
        y = PixelSNAIL(1, M_blocks=8, R_repeats=2, D_filters=64, key_size=4, value_size=32)(x)
        return y

    batch = next(ds)
    x = batch['image']
    print(x.shape)

    pixelsnail = hk.without_apply_rng(hk.transform(forward))
    if load:
        params = load_params('models/model_small_1800_72.26.pkl')
    else:
        params = pixelsnail.init(jax.random.PRNGKey(28), x)
    output = pixelsnail.apply(params, x)

    if train:
        optimizer = optax.adam(3e-4)
        opt_state = optimizer.init(params)
        
        def loss_fn(params, batch):
            x = batch['image']
            output = pixelsnail.apply(params, x)
            likelihood = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(output, x), axis=(1, 2, 3)), axis=0)
            return likelihood

        @jax.jit
        def update(params, opt_state, batch):
            likelihood, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return likelihood, params, opt_state

        @jax.jit
        def metrics_fn(params, batch):
            x = batch['image']
            output = pixelsnail.apply(params, x)
            mse = jnp.sqrt(jnp.mean( (jax.nn.sigmoid(output) - x)**2))
            likelihood = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(output, x), axis=(1, 2, 3)), axis=0)
            return {'mse': mse, 'loss': likelihood}

        losses = []
        valid_loss = []
        pbar = tqdm.trange(1000 + 1)
        for i in pbar:
            batch = next(ds)
            loss, params, opt_state = update(params, opt_state, batch)
            losses.append(loss)

            if i % 10 == 0:
                val_batch = next(val_ds)
                metrics = metrics_fn(params, val_batch)
                valid_loss.append(metrics['loss'])
                pbar.set_postfix_str(s=f'loss: {np.mean(losses[-30:]):2f}, val_loss: {np.mean(valid_loss[-3:]):2f}')

            if i % 200 == 0:
                save_params(params, name=f'{save_name}_{i}_{loss:.2f}.pkl')

    output = jax.nn.sigmoid(pixelsnail.apply(params, batch['image'])) > 0.5
    show_image_grid(np.concatenate([output[:32], batch['image'][:32]], axis=0).reshape(8, 8, *shape))

    generate_samples = make_sample_func(pixelsnail, params, x, 64)
    samples = generate_samples(jax.random.PRNGKey(42))
    show_image_grid(samples.reshape(8, 8, *shape))
    plt.show()