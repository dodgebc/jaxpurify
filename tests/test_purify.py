import jax
import jax.numpy as jnp
import jax.random as jr

import jaxpurify as jp
from jaxpurify import purify

rng = jr.key(0)

def test_params():
    @purify
    def model():
        x = jp.param(3, name="x")
        return x**2

    params = model.normal(rng)
    result = model(params)

    assert params["x"].shape == (3,)
    assert jnp.allclose(result, params["x"]**2)

def test_shapes():
    @purify
    def model():
        x = jp.param((4,3), name="x")
        y = jp.param(2, name="y")

    shapes = model.shapes()
    zeros = model.zeros()
    params = model.normal(rng)

    assert shapes["x"].shape == (4, 3)
    assert shapes["y"].shape == (2,)
    assert zeros["x"].shape == (4, 3)
    assert zeros["y"].shape == (2,)
    assert params["x"].shape == (4, 3)
    assert params["y"].shape == (2,)

def test_ravel():
    @purify(ravel=True)
    def model():
        x = jp.param((4,3), name="x")
        y = jp.param(2, name="y")

    shapes = model.shapes()
    zeros = model.zeros()
    params = model.normal(rng)

    unraveled_params = model.unravel(params)

    assert shapes.shape == (14,)
    assert zeros.shape == (14,)
    assert params.shape == (14,)
    assert unraveled_params["x"].shape == (4, 3)
    assert unraveled_params["y"].shape == (2,)

def test_fixed():
    @purify
    def model():
        x = jp.param(3, name="x")
        b = jp.fixed((2,3), name="b")
        return x + b

    params = model.normal(rng)
    fixed = model.fixed_shapes()
    result = model(params, b=1.0)

    assert fixed["b"].shape == (2, 3)
    assert jnp.allclose(result, params["x"] + 1.0)

def test_intermediates():
    @purify
    def model():
        x = jp.param(3, name="x")
        y = jp.intermediate(x**2, name="y")
        return y + 1

    params = model.normal(rng)
    intermediates = model.intermediates(params)
    assert jnp.allclose(intermediates["y"], params["x"]**2)

def test_with_custom_jvp():
    @purify
    def model():
        return jax.nn.relu(jp.param(3, name="x"))

    params = model.normal(rng)
    results = model(params)
    assert jnp.allclose(results, jax.nn.relu(params["x"]))

def test_higher_order_primitives():
    def f(x):
        a = jp.param()
        return jnp.sin(a * x)

    @purify
    def model():
        x = jp.param(3, name="x")
        y = jax.jit(jax.vmap(jax.grad(jax.grad(f))))(x)
        return jnp.sum(y)

    def model_explicit(x):
        y = jax.jit(jax.vmap(jax.grad(jax.grad(f))))(x)
        return jnp.sum(y)

    params = model.normal(rng)
    result = model(params)
    result_explicit = model_explicit(params['x'])
    assert jnp.allclose(result, result_explicit)

    result_grad = jax.grad(model)(params)
    result_grad_explicit = jax.grad(model_explicit)(params['x'])
    assert jnp.allclose(result_grad['x'], result_grad_explicit)

    