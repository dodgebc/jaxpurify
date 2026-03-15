"""
Demonstrate how to use jaxpurify to build neural networks.
This is not the targeted use case but the code turns out to be quite elegant.
One major downside is that neural networks are often trees with all arrays trainable
and so it is more natural to store the model as a callable PyTree, similar to Equinox or Flax.

"""

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

import jaxpurify as jp
from jaxpurify import purify

rng = jr.key(0)


# %% The simplest approach, unnamed parameters
def linear(x, n_in, n_out):
    w = jp.param((n_out, n_in))
    b = jp.param(n_out)
    return w @ x + b

@purify
def model():
    x = jp.fixed(16, name="x")
    x = linear(x, 16, 8)
    x = jax.nn.relu(x)
    x = linear(x, 8, 4)
    x = jax.nn.softmax(x)
    return x

print(model.shapes())
print(model(model.normal(rng), x=jnp.ones(16)))

# %% An alternative pattern for reference, now also with unique parameter names
def linear(n_in, n_out, name):
    def apply(x):
        w = jp.param((n_out, n_in), name=f"{name}_w")
        b = jp.param(n_out, name=f"{name}_b")
        return w @ x + b
    return apply

@purify
def model():
    layers = [
        linear(16, 8, name="linear_1"),
        jax.nn.relu,
        linear(8, 4, name="linear_2"),
        jax.nn.softmax,
    ]
    x = jp.fixed(16, name="x")
    for layer in layers: x = layer(x)
    return x

print(model.shapes())

# %% Factory function for building a dense network
def dense_network(dims):
    def model(x):
        n = len(dims) - 1
        for i in range(len(dims) - 1):
            if i > 0: x = jax.nn.relu(x)
            x = linear(dims[i], dims[i+1], name=f"linear_{i+1}")(x)
        return x
    return model

@purify
def model():
    x = jp.fixed(16, name="x")
    x = dense_network([16, 8, 4])(x)
    return jax.nn.softmax(x)

print(model.shapes())
# %%
