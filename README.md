# JAX Purify
A simple [Jaxpr interpreter](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html) which allows you to define models like this:

```python
@purify
def model():
    x = param(3)
    return jnp.sin(x)

params = model.normal(rng)
model(params)
```

Parameters are only mentioned once, right next to where they are used. This avoids initialization methods and object-oriented programming while retaining automatic parameter management.

Other notable features:
- `fixed` arrays for data that changes between calls but should not be be differentiated
- `intermediate` outputs for extracting internal values without restructuring the model
- `ravel` keyword for accepting parameters as a single 1d array

I wrote this tool to make it easier to read and write probabilistic models in astrophysics after getting frustrated using neural network libraries for the task. Our generative models take physical parameters to be learned and apply known transformations to produce predicted data. We often work in unit normal "whitened" parameter space, our model graphs are not trees, new custom components frequently need to be made, intermediate outputs are just as important as predicted data, and fixed arrays abound. These characteristics determined the design.

The "purifying" transformation is not a new idea, to be clear. [Haiku](https://github.com/google-deepmind/dm-haiku) transforms models into pure functions but without the features above and with side-effect issues. [Oryx](https://github.com/jax-ml/oryx) works in a similar way with a more general scope but a different interface. And there is plenty of other discussion in this direction [[1]](https://sjmielke.com/jax-purify.htm) [[2]](https://github.com/jax-ml/jax/discussions/14661).


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
    k = jnp.fft.fftfreq(n).at[0].set(jnp.inf)
    power = jnp.abs(k) ** alpha
    xi = jp.param(n, name="xi") # declare white noise parameter
    f = jnp.fft.fft(jnp.sqrt(power) * xi)
    return f.real - f.imag

@purify
def model():
    x = jnp.linspace(0, 5, 100)
    alpha = -jp.log_normal_variable("alpha", mean=3.0, sigma=0.1) # useful parameter shortcut
    f = jp.intermediate(field(x.shape[0], alpha), name="f") # intermediate return
    x_obs = jp.fixed(3, name="x_obs") # passed at evaluation time
    f_obs = jnp.interp(x_obs, x, f)
    return f_obs

# ways to get parameter dictionary
param_shapes = model.shapes()
zero_params = model.zeros()
params = model.normal(jr.key(137))

# forward pass and intermediates pass
x_obs = jnp.array([1, 2, 3])
result = model(params, x_obs=x_obs)
result_field = model.intermediates(params, x_obs=x_obs)["f"]

```

## Q&A

#### How does it work?

JAX transformations like `grad` and `vmap` work by walking through the "Jaxpr" that describes the computation and interpreting each primitive in a new light. We use this same infrastructure for `purify`, inserting parameters into the computational graph where needed during evaluation. Under the hood, `param`, `fixed`, and `intermediate` are identity primitives which exist only to trigger special handling. If the model is not purified, parameters and fixed values are just arrays of zeros. This [tutorial](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html) is a great introduction to the interpreter system in JAX.

#### Is it compatible with arbitrary JAX transformations anywhere I like?

Almost! You should be able to apply transformations both inside the model and to the resulting purified function and it will behave exactly as if the parameter was already an initialized array. The only exceptions are certain higher-order primitives like `scan`, where if you declare parameters inside the body function they will be missed. Passing them through or closing over them are totally fine.

#### What happens to Python control flow inside the model?

It works exactly like it does under `jit`. No problem unless the control flow depends on array values, in which case a tracer conversion error will be raised. By the way, the purified function can be automatically compiled with `jit=True` if desired.

#### How do you deal with duplicate parameter names when reusing a component?

There is no automatic way. For models which do not have a simple tree structure, what good names could even be assigned? One could append "_1", "_2", etc based on the order of application, but this can be confusing and subject to change in complex models. Instead, if you want named parameters in reusable components, just accept a `name` argument as shown in `examples/neural_network.py`.

#### Is this better than using callable PyTrees like Equinox?

It depends. I find initialization and other class boilerplate to be verbose and distracting, especially when compared to what is possible with `jaxpurify`. Also, fixed arrays in PyTree models must be excluded from derivatives via `stop_gradient`, Equinox-style filtering, or by manually passing them as arguments deep into the model, all of which I find unwieldy. On the other hand, if your model graph is a tree with all arrays learnable (read: neural network), it is sad to give up the convenience of a tree-like data structure which enables simple inspection and intermediate evaluation while avoiding the duplicate name issue mentioned above.

#### How is this different than Oryx?

Parameters and fixed values are declared with shape. Parameters are assumed to be unit normal and can then be transformed into other distributions. Parameter raveling and class-like `model.normal(rng)`, `model(params)`, etc are also provided. These are all surface-level ergonomics which make code shorter. On the other hand, Oryx is a much more substantial effort with more general scope and better handling of certain higher-order primitives. If it is convenient to read and write your model in Oryx, that is probably preferable.

#### Why do I need to declare the shapes of fixed values in the model?

One could imagine determining shapes from whatever arrays are passed, eliminating the need to synchronize the model. But then fixed values would need to be passed to the initialization methods, each model call would need to re-trace the function, it would be more complicated to harvest fixed value names, and the unpurified model would not have a well-defined call behavior. Instead, if you need to change any shapes in your model frequently, just make a factory function as shown in `examples/neural_network.py`.