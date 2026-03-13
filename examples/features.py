"""
Explain the basic features of jaxpurify:
 - parameter management with `shapes`, `zeros`, and `normal`
 - fixed array inputs with `fixed`
- modularity with parameterized model components
 - intermediate return values with `intermediate`
 - flat vector parameters with `ravel=True` and `unravel`

"""

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

import jaxpurify as jp
from jaxpurify import purify

rng = jr.key(0)


# %% Basic model with random parameters
@purify
def model():
    x = jp.param(3, name="x")
    return x**2

params = model.normal(rng)
result = model(params)

print("Params:", params)
print("Result:", result)

# %% Deep model with multiple components
def linear(x, n_in, n_out):
    w = jp.param((n_out, n_in))
    b = jp.param(n_out)
    return w @ x + b

@purify
def model():
    x = jp.fixed(32, name="x")
    x = linear(x, 32, 16)
    x = jax.nn.relu(x)
    x = linear(x, 16, 8)
    x = jax.nn.relu(x)
    x = linear(x, 8, 4)
    x = jax.nn.log_softmax(x)
    return x

params = model.normal(rng)
result = model(params, x=jnp.ones(32))

print("Params:", params)
print("Result:", result)


# %% Demonstration of `shapes`, `zeros`, `fixed`, and `intermediates` convenience methods
@purify
def model():
    a = jnp.array([[1, 2, 3]])
    x = jp.param(3, name="x")
    b = jp.fixed(name="b")
    y = jp.intermediate(a * x, name="prod") + b
    return y

param_shapes = model.shapes()
zero_params = model.zeros()
fixed = model.fixed()
result = model(zero_params, b=2.0)
intermediates = model.intermediates(zero_params, b=2.0)

print("Param shapes:", param_shapes)
print("Zero params:", zero_params)
print("Fixed:", fixed)
print("Result:", result)
print("Intermediates:", intermediates)


# %% Demonstration of `ravel=True` and `unravel` functionality
@purify(ravel=True)
def model():
    x = jp.param(2)
    y = jp.param((2, 2))
    z = jnp.sum(x * y)
    return z

flat_params = model.normal(rng)
unraveled_params = model.unravel(flat_params)
result = model(flat_params)

print("Flat params:", flat_params)
print("Unraveled params:", unraveled_params)
print("Result:", result)
# %%
