# JAX Purify
A simple JAX *interpreter* which allows you to define models like this:

```python
@purify
def model():
    x = param(3, name="x")
    return jnp.sin(x)

params = model.normal(rng)
model(params)
```

Why would you want to do this? Well, you might not. But this approach has several desirable properties:

1. Arbitrary operations can be applied in the same way irrespective of whether the arguments are parameters, fixed arrays, or the output of other model components.
2. Parameter names and shapes are only written in one place, right next to where they are used.

Consider the equivalent model written explicitly as a function. Notice that the parameter shapes are specified far from where they are used, and the names of the parameters must be written twice. In deep models it becomes cumbersome to pass parameters around and keep everything synchronized as they are deleted, renamed, or change shapes. It is thus not very popular to write complex models with functions alone.

```python
def model(params):
    x = params["x"]
    return jnp.sin(x)

params = {"x": jr.normal(rng, 3)}
model(params)
```

A familiar alternative is to organize models into classes, similar to PyTorch, Equinox, Flax, etc. The advantage is that model components keep track of their own parameters. For models which are a stack 

```python
@register_dataclass
@dataclass
class Model:
    x: Array

    def __init__(self, rng):
        self.x = jr.normal(rng, 3)

    def __call__(self):
        return jnp.sin(self.x)
```