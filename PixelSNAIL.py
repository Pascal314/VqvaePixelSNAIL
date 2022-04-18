import typing
from typing import Tuple
import haiku as hk
import jax
from math import prod
import jax.numpy as jnp
import numpy as np

class ResidualBlock(hk.Module):
    def __init__(self, filters, repeats=1):
        super().__init__()
        self.filters = filters
        self.repeats = repeats

    def __call__(self, x):
        for _ in range(self.repeats):
            r = jax.nn.elu(x)
            r = hk.Conv2D(self.filters, (2, 2), padding=hk.pad.causal)(r)
            r = jax.nn.elu(r)
            gate = hk.Conv2D(self.filters, (2, 2), padding=hk.pad.causal)(r)
            gate = jax.nn.sigmoid(gate)
            r = hk.Conv2D(self.filters, (2, 2), padding=hk.pad.causal)(r)
            r = r * gate
            x = r + x
        return x


def masked_softmax(x, mask, axis=-1):
    x = x * mask
    x_max = jnp.max(x, axis, where=mask, keepdims=True, initial=0)
    unnormalized = jnp.exp(x - jax.lax.stop_gradient(x_max))
    return unnormalized * mask / (jnp.sum(unnormalized, axis, where=mask, keepdims=True) + 1e-16)

def causal_attention_mask(n):
    mask = jnp.tril(jnp.ones((n, n), dtype=np.float32), -1)
    return mask.reshape(1, n, n) # This mask is the same across the batch dim.

def attention_lookup(Q, K, V):
    bs, *features_size, ks = Q.shape
    # reshape into (batch size, all_features, key_size)
    # e.g. mnist -> (batch size, 28 * 28, 1)
    fs = prod(features_size)
    # Batched matmul
    logits = jnp.matmul(Q.reshape(bs, fs, ks), K.reshape(bs, fs, ks).swapaxes(1, 2)) / np.sqrt(ks) # this is now (bs, fs, fs)
    probs = masked_softmax(logits, mask=causal_attention_mask(logits.shape[1])) # this is now still (bs, fs, fs)
    outputs = jnp.matmul(probs, V.reshape(bs, fs, V.shape[-1])) # this is (bs, fs, vs)
    return outputs.reshape(bs, *features_size, V.shape[-1]) # original shape except for different output channels.

class AttentionBlock(hk.Module):
    def __init__(self, key_size, value_size):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

    def __call__(self, x, extra_input=None):
        Q = hk.Conv2D(self.key_size, 1)(x)
        if extra_input is not None:
            x = jnp.concatenate([x, extra_input], axis=-1)
        K = hk.Conv2D(self.key_size, 1)(x)
        V = hk.Conv2D(self.value_size, 1)(x)
        return attention_lookup(Q, K, V)

def strict_causal_padding(effective_filter_size: int) -> Tuple[int, int]:
    return (effective_filter_size, -1)

def strict_causal_padding(effective_filter_size: int) -> Tuple[int, int]:
    return (effective_filter_size, -1)

class PixelSNAIL(hk.Module):
    def __init__(self, output_size, M_blocks=12, R_repeats=4, D_filters=256, key_size=16, value_size=128):
        super().__init__()
        self.M_blocks = M_blocks
        self.R_repeats = R_repeats
        self.D_filters = D_filters
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size

    def __call__(self, x):
        _input = hk.Conv2D(self.D_filters, (2, 2), padding=strict_causal_padding)(x)
        block_input = _input
        for i in range(self.M_blocks):
            r_output = ResidualBlock(self.D_filters, repeats=self.R_repeats)(block_input)

            # the attention path
            attention_path = AttentionBlock(self.key_size, self.value_size)(r_output, x)
            attention_path = jax.nn.elu(attention_path)
            attention_path = hk.Conv2D(self.D_filters, (1, 1))(attention_path)

            #the other path
            conv_path = jax.nn.elu(r_output)
            conv_path = hk.Conv2D(self.D_filters, (1, 1))(conv_path)
            conv_path = jax.nn.elu(conv_path)

            block_output = conv_path + attention_path
            block_output = jax.nn.elu(block_output)
            block_output = hk.Conv2D(self.D_filters, (1, 1))(block_output)
            block_input = jax.nn.elu(block_output) + _input

        output = hk.Conv2D(self.output_size, (1, 1))(block_input)
        return output

def make_sample_func(net, params, x, n_samples):
    x = net.apply(params, x)
    shape = (n_samples, ) + x.shape[1:-1]
    n_classes = x.shape[-1]

    if n_classes > 1:
        @jax.jit
        def sample_pixel(x, key, i, j):
            logits = net.apply(params, x)[:, i, j, :]
            sampled = jax.random.categorical(key, logits)
            return sampled
    else:
        @jax.jit
        def sample_pixel(x, key, i, j):
            logits = net.apply(params, x.reshape(*x.shape, 1))[:, i, j, :]
            sampled = jax.random.bernoulli(key, jax.nn.sigmoid(logits))
            return sampled

    def generate_samples(rng_key):
        x = np.zeros(shape)
        for i in range(shape[1]):
            for j in range(shape[2]):
                new_key, rng_key = jax.random.split(rng_key)
                sampled = sample_pixel(x, new_key, i, j)
                x[:, i, j] = sampled.squeeze()
        return x
    return generate_samples    