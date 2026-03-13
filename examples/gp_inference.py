"""
Infer a field from noisy sparse observations using a Gaussian process prior.
The measurement is linear and the noise is Gaussian so the posterior has an analytic solution.
We can approximately solve the required matrix equations using conjugate gradient iteration, avoiding expensive dense matrices.

Specifically, given a model d = R s + n where n ~ N(0, N) is noise and s ~ N(0, I) is unit normal a priori ("whitened" parameters),
we can solve for posterior mean m and mean-subtracted posterior samples z using

    m = (I + R^T N^-1 R)^-1 (R_T N^-1 d)
    z = (I + R^T N^-1 R)^-1 (R_T N^-1 q + p)
    
where q ~ N(0, I) and p ~ N(0, I) are independent noise samples of data and parameter dimension, respectively. It is easy to show
that <z z^T> = (I + R^T N^-1 R)^-1, as desired.

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


# %% Define model
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


# %% Simulate data
rng, k1, k2, k3 = jr.split(rng, 4)
grid = jnp.linspace(0, 20, 1000)
x_obs = jr.uniform(k1, 100, minval=grid.min(), maxval=grid.max())
y_true = jnp.sin(x_obs)
noise_std = jnp.sqrt(jnp.abs(y_true))/2
noise = noise_std * jr.normal(k2, x_obs.shape)
y_obs = y_true + noise

# %% Solve for posterior mean using CG
R = partial(model, grid=grid, x_obs=x_obs)
R_T = lambda d: jax.linear_transpose(R, model.shapes())(d)[0]
N_inv = lambda x: x / noise_std**2
M = lambda x: R_T(N_inv(R(x))) + x
mean = cg(M, R_T(N_inv(y_obs)), x0=model.zeros(), maxiter=100)[0]
field_mean = model.intermediates(mean, grid=grid, x_obs=x_obs)["y"]

# %% Solve for posterior samples using CG and sampling trick
n_samples = 5
rng, k1, k2 = jr.split(rng, 3)
data_noise = jr.normal(k1, (n_samples, *y_obs.shape))
prior_noise = jr.normal(k2, (n_samples, *mean.shape))
samples = mean + jax.vmap(partial(cg, M, maxiter=100))(prior_noise + jax.vmap(R_T)(data_noise), prior_noise)[0]
field_samples = jax.vmap(partial(model.intermediates, grid=grid, x_obs=x_obs))(samples)["y"]

# %% Plot results!
plt.plot(grid, jnp.sin(grid), c='k', alpha=0.5)
plt.errorbar(x_obs, y_obs, yerr=noise_std, fmt="o", c='k', alpha=0.5, ms=2, lw=1)
plt.plot(grid, field_mean, c='C0')
plt.plot(grid, field_samples.T, alpha=0.5, c='C0')
plt.show()
# %%
