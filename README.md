# JAX Purify
A simple [Jaxpr interpreter](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html) which allows you to define models like this:

```python
@purify
def model():
    x = param(3, name="x")
    return jnp.sin(x)

params = model.normal(rng)
model(params)
```

Parameters are only mentioned once, right next to where they are used. This avoids initialization methods and object-oriented programming while retaining automatic parameter management.

Other notable features:
- `fixed` arrays for data that changes between calls but should not be be differentiated
- `intermediate` outputs for extracting internal values without restructuring your model
- `ravel` keyword for accepting parameters as a single 1d array

The idea is basically the same as [Haiku](https://github.com/google-deepmind/dm-haiku) but without the module abstraction and adding the features above. The implementation is also different and allows arbitrary JAX transformations inside your model, which cause side-effecting [issues](https://dm-haiku.readthedocs.io/en/latest/notebooks/transforms.html) in Haiku.

I wrote this tool to make it easier to read and write probabilistic models in astrophysics after getting frustrated using neural network libraries for the task. We often work in unit normal "whitened" parameter space, models are not trees but graphs, new custom components frequently need to be made, intermediate outputs are just as important as predicted data, and fixed arrays abound. These characteristics determined the design.

## Installation

To install, use `pip`. The only dependency is `jax`.

```
python -m pip install jaxpurify
```

## An Instructive Example

Suppose we want to use a Gaussian process to model a continuous field and read it out at several measurement locations. This example illustrates all of the core features of `jaxpurify` with the exception of parameter raveling. A complete inference demonstration is provided in `examples/gaussian_process.py`. See also `examples/features.py` for a more gradual introduction.

```python
import jax.numpy as jnp
import jax.random as jr
import jaxpurify as jp
from jaxpurify import purify # aesthetics

def field(n, alpha):
    # spectral index alpha, taken as an argument
    k = jnp.fft.fftfreq(n).at[0].set(jnp.inf)
    power = jnp.abs(k) ** alpha

    # xi will always be white noise, so declare here
    xi = jp.param(n, name="xi")
    f = jnp.fft.fft(jnp.sqrt(power) * xi)
    return f.real - f.imag

@purify
def model():
    # fixed model grid is closed over
    x = jnp.linspace(0, 5, 100)

    # shortcut for a transformed white noise parameter
    alpha = -jp.log_normal_variable("alpha", mean=3.0, sigma=0.1)
    
    # f will be returned as an intermediate
    f = jp.intermediate(field(x.shape[0], alpha), name="f")

    # fixed x_obs will be passed to model at evaluation time
    x_obs = jp.fixed(3, name="x_obs")

    f_obs = jnp.interp(x_obs, x, f)
    return f_obs

# stuff you can do with parameters
param_shapes = model.shapes()
zero_params = model.zeros()
params = model.normal(jr.key(67))

# forward pass and intermediates pass
x_obs = jnp.array([1, 2, 3])
result = model(params, x_obs=x_obs)
result_field = model.intermediates(params, x_obs=x_obs)["f"]

```

## Q&A

### How does it work?

JAX transformations like `grad` and `vmap` work by walking through the "Jaxpr" that describes the computation and interpreting each primitive in a new light. We use this same infrastructure for `purify`, inserting parameters into the computational graph where needed during evaluation. Under the hood, `param`, `fixed`, and `intermediate` are identity primitives which exist only to trigger special handling. If the model is not purified, parameters and fixed values are just arrays of zeros. This [tutorial](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html) is a great introduction to the interpreter system in JAX.

### Is it compatible with arbitrary JAX transformations anywhere I like?

Yes! You should be able to apply arbitrary transformations both inside the model and to the resulting purified function. That being said, `jaxpurify` uses JAX internals which are not documented or promised to be stable, so if something breaks please open an issue.

### What happens to Python control flow inside the model?

It will be executed assuming zeros for parameters and fixed values. As a general rule, don't do anything that would prevent you from applying `jit` to the purified model. By the way, this can be automatically done with `jit=True` if desired.

### How do you deal with duplicate parameter names when reusing a component?

There is no automatic way. For models which do not have a simple tree structure, what good names could even be assigned? One could append "_1", "_2", etc based on the order of application, but this can be confusing and subject to change in complex models. Instead, if you want named parameters in reusable components, just accept a `name` argument as shown in `examples/neural_network.py`.

### Is this better than using callable PyTrees like Equinox?

It depends. I find initialization and other class boilerplate to be verbose and distracting, especially when compared with what is possible with `jaxpurify`. Also, fixed arrays in PyTree models must be excluded from derivatives via `stop_gradient`, Equinox-style filtering, or by manually passing them as arguments deep into the model, all of which I find unwieldy. On the other hand, if your model is a tree, it is hard to give up the convenience of a tree-like data structure which enables simple inspection and intermediate evaluation while avoiding the duplicate name issue mentioned above. If your model is a graph, there is little to be done.

### Why do I need to declare the shapes of fixed values in the model?

One could imagine determining shapes from whatever arrays are passed, eliminating the need to synchronize the model. But then fixed values would need to be passed to the initialization methods, each model call would need to re-trace the function, it would not be possible to harvest fixed value names, and the unpurified model would not have a well-defined call behavior. Instead, if you need to change model shapes frequently, just make a factory function.