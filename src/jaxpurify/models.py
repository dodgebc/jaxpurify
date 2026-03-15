import jax
import jax.numpy as jnp
import jax.random as jr

from .purify import param, intermediate


def normal_variable(name, mean=0.0, sigma=1.0):
    """Shortcut to create a scalar normal random variable which is both a parameter and intermediate value. Common for learned hyperparameters."""
    return intermediate(normal(param(name=name), mean=mean, sigma=sigma), name=name)


def log_normal_variable(name, mean=1.0, sigma=1.0):
    """Shortcut to create a scalar log-normal random variable which is both a parameter and intermediate value. Common for learned hyperparameters."""
    return intermediate(log_normal(param(name=name), mean=mean, sigma=sigma), name=name)


def uniform_variable(name, low=0.0, high=1.0):
    """Shortcut to create a scalar uniform random variable which is both a parameter and intermediate value. Common for learned hyperparameters."""
    return intermediate(uniform(param(name=name), low=low, high=high), name=name)


def normal(x, *, mean, sigma):
    """Transform a standard normal variable `x` into a normal variable with specified mean and standard deviation."""
    return mean + sigma * x


def log_normal(x, *, mean, sigma):
    """Transform a standard normal variable `x` into a log-normal variable with specified mean and standard deviation. Note that the mean must be positive."""
    mu = jnp.log(mean**2 / jnp.sqrt(mean**2 + sigma**2))
    sigma = jnp.sqrt(jnp.log(1 + sigma**2 / mean**2))
    return jnp.exp(mu + x * sigma)


def uniform(x, *, low=0.0, high=1.0):
    """Transform a standard normal variable `x` into a uniform variable with specified lower and upper bounds."""
    uniform = (1 + jax.lax.erf(x / jnp.sqrt(2))) / 2
    return low + (high - low) * uniform


def zeros_like(x):
    """Fill a pytree with zeros."""
    leaves, treedef = jax.tree_util.tree_flatten(x)
    zero_leaves = [jnp.zeros_like(leaf) for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, zero_leaves)


def normal_like(rng, x):
    """Fill a pytree with unit normal values."""
    leaves, treedef = jax.tree_util.tree_flatten(x)
    normal_leaves = [jr.normal(rng, leaf.shape, leaf.dtype) for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, normal_leaves)
