"""
Instructive examples of how to use jaxpurify.

 1. Basic model with random parameters
 2. Demonstration of `shapes`, `zeros`, `fixed`, and `intermediates` convenience methods
 3. Demonstration of `ravel=True` and `unravel` functionality
 4. Complex mock example with function calls, log-normal and uniform variables, and higher-order primitives
 5. Bayesian inference application with a linear Gaussian process model

"""

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.sparse.linalg import cg
from functools import partial
import matplotlib.pyplot as plt

import jaxpurify as jp
from jaxpurify import purify

rng = jr.key(0)


# %% 1. Basic model with random parameters
@purify
def model():
    x = jp.param(3, name="x")
    return x**2

params = model.normal(rng)
result = model(params)

print("Params:", params)
print("Result:", result)


# %% 2. Demonstration of `shapes`, `zeros`, `fixed`, and `intermediates` convenience methods
@purify
def model():
    a = jnp.array([[1, 2, 3]])
    x = jp.param(3, name="x")
    b = jp.fixed(name="b")
    y = jp.intermediate(a * x, name="prod") + b
    return y

param_shapes = model.shapes()
zero_params = model.zeros()
zero_fixed = model.fixed()
result = model(zero_params, {"b": 2.0})
intermediates = model.intermediates(zero_params, {"b": 2.0})

print("Param shapes:", param_shapes)
print("Zero params:", zero_params)
print("Zero fixed:", zero_fixed)
print("Result:", result)
print("Intermediates:", intermediates)


# %% 3. Demonstration of `ravel=True` and `unravel` functionality
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

# %% 4. Complex mock example with many functions, log-normal and uniform variables, and higher-order primitives
def chop(vegetable, slices):
    return (vegetable / slices) * jnp.ones(slices)

def boil(ingredients, *, time):
    kernel = jnp.exp(-(jnp.linspace(-60/(time+1), 60/(time+1), 15) ** 2))
    return jnp.convolve(ingredients, kernel, mode="same")

def sautee(ingredients, *, time):
    scale = jnp.exp(time / 5) / jnp.sqrt(ingredients.size)
    return scale * jnp.cumsum(ingredients)

def prepare_aromatics(onions, carrots, celery, *, time_onions, time_together):
    chop_multiple = jax.vmap(chop, in_axes=(0, None))
    onions = chop_multiple(onions, 30).flatten()
    carrots = chop_multiple(carrots, 5).flatten()
    celery = chop_multiple(celery, 5).flatten()
    onions = sautee(onions, time=time_onions)
    aromatics = sautee(jnp.concatenate([onions, carrots, celery]), time=time_together)
    return aromatics

def eat(food):
    return jnp.sum(food**2)

@purify
def dinner():
    carrots = jnp.cos(jp.param(5, name="carrots"))
    celery = jnp.sin(jp.param(5, name="celery"))
    onions = jnp.tanh(jp.param(2, name="onions"))

    aromatics = prepare_aromatics(
        onions, carrots, celery,
        time_onions=jp.LogNormalVariable("onion_time", mean=10, sigma=1),
        time_together=jp.LogNormalVariable("aromatics_time", mean=3, sigma=1),
    )
    aromatics = jp.intermediate(aromatics, name="aromatics")

    seasoning = sum([
        jp.LogNormalVariable("salt", mean=10, sigma=2),
        jp.LogNormalVariable("thyme", mean=3, sigma=1),
        jp.UniformVariable("bay_leaf", low=2, high=3),
    ])

    lentils = jp.log_normal(jp.param(300, name="lentils"), mean=1.0, sigma=0.1)
    lentils = jax.jvp(jnp.sin, (lentils,), (lentils,))[1]
    soup = boil(
        seasoning * jnp.concatenate([lentils, aromatics]),
        time=jp.UniformVariable("boil_time", low=20, high=30),
    )
    soup += jp.LogNormalVariable("lemon", mean=0.4, sigma=0.1)
    soup = jp.intermediate(soup, name="soup")

    return eat(soup)

params = dinner.normal(rng)
result = dinner(params)
intermediates = dinner.intermediates(params)
lots_of_soup = jax.vmap(dinner.normal)(jr.split(rng, 10))
grads = jax.jit(jax.vmap(jax.grad(dinner)))(lots_of_soup)

# %% 5. Bayesian inference application with a linear Gaussian process model
def field(xi, pad=10):
    n = xi.shape[0]
    k = jnp.fft.fftfreq(n, d=1.0/n)
    p = jnp.power(jnp.abs(k), -3.0).at[0].set(1.0)
    f = jnp.fft.fft(jnp.sqrt(p) * xi)
    f = (f.real - f.imag)
    return f[pad:n-pad]

@purify(ravel=True)
def model():
    pad = 10
    grid = jp.fixed(1000, name="grid")
    xi = jp.param(grid.shape[0] + 2*pad, name="xi")
    y = jp.intermediate(field(xi, pad=pad), name="y")
    x_obs = jp.fixed(100, name="x_obs")
    y_obs = jnp.interp(x_obs, grid, y)
    return y_obs

rng, k1, k2, k3 = jr.split(rng, 4)

# Simulate some data
grid = jnp.linspace(0, 20, 1000)
x_obs = jr.uniform(k1, 100, minval=grid.min(), maxval=grid.max())
fixed = {"grid": grid, "x_obs": x_obs}
y_true = jnp.sin(x_obs)
noise_std = jnp.sqrt(jnp.abs(y_true))/2
noise = noise_std * jr.normal(k2, x_obs.shape)
y_obs = y_true + noise

# Solve for posterior mean using CG
R = partial(model, fixed=fixed)
R_T = lambda d: jax.linear_transpose(R, model.shapes())(d)[0]
N_inv = lambda x: x / noise_std**2
M = lambda x: R_T(N_inv(R(x))) + x
mean = cg(M, R_T(N_inv(y_obs)), x0=model.zeros(), maxiter=100)[0]
field_mean = model.intermediates(mean, fixed)["y"]

# Solve for posterior samples using CG and sampling trick
n_samples = 5
rng, k1, k2 = jr.split(rng, 3)
data_noise = jr.normal(k1, (n_samples, *y_obs.shape))
prior_noise = jr.normal(k2, (n_samples, *mean.shape))
samples = mean + jax.vmap(partial(cg, M, maxiter=100))(prior_noise + jax.vmap(R_T)(data_noise), prior_noise)[0]
field_samples = jax.vmap(model.intermediates, in_axes=(0, None))(samples, fixed)["y"]

# Plot results!
plt.plot(grid, jnp.sin(grid), c='k', alpha=0.5)
plt.errorbar(x_obs, y_obs, yerr=noise_std, fmt="o", c='k', alpha=0.5, ms=2, lw=1)
plt.plot(grid, field_mean, c='C0')
plt.plot(grid, field_samples.T, alpha=0.5, c='C0')
plt.show()

