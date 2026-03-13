"""
Demonstrate how to build a deep, complex model with multiple reusable components.
In this case, all parameters are defined in a centralized "orchestrator" function,
and passed as needed to various components. It is just as easy to define parameters
in the components themselves as shown in other examples, depending on what makes the
most sense for the application.
"""

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

import jaxpurify as jp
from jaxpurify import purify

rng = jr.key(0)


# %% Define model
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

def make_soup(lentils, aromatics, seasoning, lemon, *, boil_time):
    soup = boil(seasoning * jnp.concatenate([lentils, aromatics]), time=boil_time)
    soup += lemon
    return soup

def eat(food):
    return jnp.sum(food**2)

@purify
def dinner():
    carrots = jnp.cos(jp.param(5, name="carrots"))
    celery = jnp.sin(jp.param(5, name="celery"))
    onions = jnp.tanh(jp.param(2, name="onions"))
    aromatics = prepare_aromatics(
        onions, carrots, celery,
        time_onions=jp.log_normal_variable("onion_time", mean=10, sigma=1),
        time_together=jp.log_normal_variable("aromatics_time", mean=3, sigma=1),
    )
    aromatics = jp.intermediate(aromatics, name="aromatics")

    seasoning = sum([
        jp.log_normal_variable("salt", mean=10, sigma=2),
        jp.log_normal_variable("thyme", mean=3, sigma=1),
        jp.uniform_variable("bay_leaf", low=2, high=3),
    ])

    lentils = jp.log_normal(jp.param(300, name="lentils"), mean=1.0, sigma=0.1)
    lentils = jax.jvp(jnp.sin, (lentils,), (lentils,))[1]

    soup = make_soup(
        lentils, aromatics, seasoning,
        jp.log_normal_variable("lemon", mean=0.4, sigma=0.1),
        boil_time=jp.uniform_variable("boil_time", low=20, high=30),
    )
    soup = jp.intermediate(soup, name="soup")

    return eat(soup)

# %% Run model
params = dinner.normal(rng)
result = dinner(params)
intermediates = dinner.intermediates(params)
lots_of_soup = jax.vmap(dinner.normal)(jr.split(rng, 10))
grads = jax.jit(jax.vmap(jax.grad(dinner)))(lots_of_soup)
