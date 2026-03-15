"""
Explain the basic features of jaxpurify:
 - parameter management with `shapes`, `zeros`, and `normal`
 - modularity with parameterized model components
 - fixed array inputs with `fixed`
 - intermediate return values with `intermediate`
 - flat vector parameters with `ravel=True` and `unravel`
- shortcut for `normal`, `uniform`, and `log_normal` variables

"""

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

import jaxpurify as jp
from jaxpurify import purify

rng = jr.key(0)

# %% Basic model with parameter shapes, zeros, and normal
@purify
def model():
    x = jp.param(3, name="x")
    return x**2

param_shapes = model.shapes()
zero_params = model.zeros()
params = model.normal(rng)
result = model(params)

print("Param shapes:", param_shapes)
print("Zero params:", zero_params)
print("Normal Params:", params)
print("Result:", result)

# %% Parameters can be defined in any function called by the model
def width():
    return jp.param(name="w")

def height():
    return jp.param(name="h")

@purify
def model():
    return jnp.sqrt(width()**2 + height()**2)

params = model.normal(rng)
result = model(params)
print("Params:", params)
print("Result:", result)

# %% Fixed arrays can be closed over or passed as keyword arguments
@purify
def model():
    a = jnp.array(1.0)
    b = jp.fixed(name="b")
    return a + b

fixed_shapes = model.fixed_shapes()
result = model(None, b=2.0)
print("Fixed shapes:", fixed_shapes)
print("Result:", result)

# %% Intermediate values can be returned and accessed by name
@purify
def model():
    x = jp.param(3, name="x")
    y = jp.intermediate(x**2, name="y")
    return jnp.sqrt(jnp.sum(y))

params = model.normal(rng)
intermediates = model.intermediates(params)
result = model(params)
print("Intermediates:", intermediates)
print("Result:", result)

# %% Parameters can be flattened into a single 1d array
@purify(ravel=True)
def model():
    x = jp.param(2)
    y = jp.param((2, 2))
    return jnp.sum(x * y)

flat_params = model.normal(rng)
unraveled_params = model.unravel(flat_params)
result = model(flat_params)

print("Flat params:", flat_params)
print("Unraveled params:", unraveled_params)
print("Result:", result)

# %% Shortcuts for common variable types
@purify
def model():
    a = jp.normal_variable("a", mean=1.0, sigma=2.0)
    b = jp.log_normal_variable("b", mean=1.0, sigma=0.5)
    c = jp.uniform_variable("c", low=-1.0, high=1.0)
    return a, b, c

params = model.normal(rng)
result = model(params)
intermediates = model.intermediates(params)
print("Params:", params)
print("Result:", result)
print("Intermediates:", intermediates)
# %%
