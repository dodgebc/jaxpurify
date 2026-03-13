import jax
import jax.numpy as jnp
import jax.random as jr

from jax.extend import core
from jax.core import ShapedArray
from jax._src.util import safe_map
from jax.interpreters import batching, mlir, ad
from jax.flatten_util import ravel_pytree

from functools import partial

rng = jr.key(0)

param_p = core.Primitive("param")
fixed_p = core.Primitive("fixed")
intermediate_p = core.Primitive("intermediate")


def param(shape=(), *, name=None, dtype=None):
    """
    Marker for a parameter in a model passed to `purify`.
    These can be initialized with `model.zeros` or `model.normal`.
    The `name` must be unique among parameters in the model, if `None` is provided the object id will be used.
    """
    x = jnp.zeros(shape, dtype)
    return param_p.bind(x, name=name)


def fixed(shape=(), *, name, dtype=None):
    """
    Marker for a fixed value in a model passed to `purify`.
    Fixed values can often just be closed over, but this allows them to be passed as an argument instead.
    The `name` is required and must be unique among fixed values in the model.
    """
    x = jnp.zeros(shape, dtype)
    return fixed_p.bind(x, name=name)


def intermediate(x, *, name):
    """
    Marker for an intermediate value to be returned by a model passed to `purify` when called with `model.intermediates`.
    The `name` is required and must be unique among intermediates in the model.
    """
    return intermediate_p.bind(x, name=name)


def trivialize(primitive):
    """
    Register an "identity" primitive which exists solely as a marker in the jaxpr for `purify`.
    This could be useful for users who want to define markers for their own custom version of `purify`.
    """

    def abstract_eval(x, name=None):
        return ShapedArray(x.shape, x.dtype)

    def impl(x, name=None):
        return x

    def batch(args, batch_dims, name=None):
        return primitive.bind(args[0], name=name), batch_dims[0]

    def lowering(ctx, x, name=None):
        return [x]

    def transpose(xt, x, name=None):
        return xt,

    def jvp(primals, tangents, name=None):
        return primitive.bind(primals[0], name=name), tangents[0]

    primitive.def_impl(impl)
    primitive.def_abstract_eval(abstract_eval)
    batching.primitive_batchers[primitive] = batch
    mlir.register_lowering(primitive, lowering)
    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose


trivialize(param_p)
trivialize(fixed_p)
trivialize(intermediate_p)


def _eqn_name(eqn):
    name = eqn.params["name"]
    if name is None:
        name = str(id(eqn.outvars[0]))
    return name

def _set_eqn(dictionary, eqn, val):
    name = _eqn_name(eqn)
    if name in dictionary:
        raise ValueError(f"Duplicate name: {name}")
    dictionary[name] = val

def purify(model=None, *, ravel=False):
    """
    Transform a JAX function which takes no parameters into function which accepts parameters, optionally fixed data, and can return intermediate values.
    These behaviors are controlled by the use of `param`, `fixed`, and `intermediate` primitives in the original function.

    The returned function should be called as `model(params, **fixed)`.

    The returned function also has the following attributes:
    - `shapes()`: returns a dictionary of parameter shapes and dtypes as `jax.ShapeDtypeStruct`.
    - `zeros()`: returns a dictionary of parameters initialized to zero (or a 1d array if `ravel=True`)
    - `normal(rng)`: returns a dictionary of parameters initialized with unit normal values (or a 1d array if `ravel=True`).
    - `fixed()`: returns a dictionary of fixed value shapes and dtypes as `jax.ShapeDtypeStruct`.
    - `intermediates(params, **fixed)`: returns a dictionary of intermediate values computed during evaluation.
    - `unravel`: if `ravel=True`, this is a function that unravels the raveled parameter vector back into the original parameter structure.

    Trivial example (see demo.py for more):
    ```python
    @purify
    def model():
        x = param(3, name="x")
        return x**2

    x = model.normal(rng)
    model(x)
    ```
    """

    if model is None:
        return partial(purify, ravel=ravel)

    jaxpr = jax.make_jaxpr(model)()

    # Extract parameter structure for raveling
    unravel = None
    if ravel:
        zero_params = {}
        for eqn in jaxpr.jaxpr.eqns:
            if eqn.primitive is param_p:
                name = _eqn_name(eqn)
                if name in zero_params:
                    raise ValueError(f"Duplicate parameter: {name}")
                zero_params[name] = jnp.zeros_like(eqn.outvars[0].aval)
        _, unravel = ravel_pytree(zero_params)

    # Parameter structure function
    def shapes():
        """Return a dictionary of `param` arrays as `jax.ShapeDtypeStruct`."""
        params = {}
        for eqn in jaxpr.jaxpr.eqns:
            if eqn.primitive is param_p:
                _set_eqn(params, eqn, jax.ShapeDtypeStruct(eqn.outvars[0].aval.shape, eqn.outvars[0].aval.dtype))
                # outvar = eqn.outvars[0]
                # params[_eqn_name(eqn)] = jax.ShapeDtypeStruct(outvar.aval.shape, outvar.aval.dtype)
        if ravel:
            total_size = sum(param.size for param in params.values())
            cast_type = jnp.result_type(*[param.dtype for param in params.values()])
            return jax.ShapeDtypeStruct((total_size,), cast_type)
        return params

    # Zero initialization function
    def zeros():
        """Return a dictionary of `param` arrays initialized to zero."""
        params = {}
        for eqn in jaxpr.jaxpr.eqns:
            if eqn.primitive is param_p:
                _set_eqn(params, eqn, jnp.zeros_like(eqn.outvars[0].aval))
                # outvar = eqn.outvars[0]
                # if _eqn_name(eqn) in params:
                #     raise ValueError(f"Duplicate parameter: {_eqn_name(eqn)}")
                # params[_eqn_name(eqn)] = jnp.zeros(outvar.aval.shape, outvar.aval.dtype)
        if ravel:
            params, _ = ravel_pytree(params)
        return params

    # Random initialization function
    def normal(rng):
        """Return a dictionary of `param` arrays initialized with unit normal values."""
        params = {}
        for eqn in jaxpr.jaxpr.eqns:
            if eqn.primitive is param_p:
                rng, subkey = jr.split(rng)
                _set_eqn(params, eqn, jr.normal(subkey, eqn.outvars[0].aval.shape, eqn.outvars[0].aval.dtype))
                # outvar = eqn.outvars[0]
                # if _eqn_name(eqn) in params:
                #     raise ValueError(f"Duplicate parameter: {_eqn_name(eqn)}")
                # params[_eqn_name(eqn)] = jr.normal(subkey, outvar.aval.shape, outvar.aval.dtype)
        if ravel:
            params, _ = ravel_pytree(params)
        return params

    # Function to make fixed values full of zeros
    def fixed():
        """Return a dictionary of `fixed` arrays as `jax.ShapeDtypeStruct`. These should be passed as kwargs to model evaluation."""
        fixed_vals = {}
        for eqn in jaxpr.jaxpr.eqns:
            if eqn.primitive is fixed_p:
                _set_eqn(fixed_vals, eqn, jax.ShapeDtypeStruct(eqn.outvars[0].aval.shape, eqn.outvars[0].aval.dtype))
                # outvar = eqn.outvars[0]
                # if _eqn_name(eqn) in fixed_vals:
                #     raise ValueError(f"Duplicate fixed parameter: {_eqn_name(eqn)}")
                # fixed_vals[_eqn_name(eqn)] = jax.ShapeDtypeStruct(outvar.aval.shape, outvar.aval.dtype)
        return fixed_vals

    # Pure forward pass
    def forward(params, **fixed):
        """Evaluate model with params and (optionally) fixed values in the structure returned by `zeros`/`normal` and `fixed`, respectively."""
        if unravel:
            params = unravel(params)
        env = {}

        def read(v):
            return v.val if type(v) is core.Literal else env[v]

        def write(v, val):
            env[v] = val

        # Bind args and consts to environment
        safe_map(write, jaxpr.constvars, jaxpr.consts)

        # Loop through equations and evaluate
        for eqn in jaxpr.jaxpr.eqns:
            # Load parameter or bind primitive
            if eqn.primitive is param_p:
                if _eqn_name(eqn) not in params:
                    raise ValueError(f"Model has parameter {_eqn_name(eqn)} but no value was provided.")
                outvals = params[_eqn_name(eqn)]
            elif eqn.primitive is fixed_p:
                if _eqn_name(eqn) not in fixed:
                    raise ValueError(f"Model has fixed parameter {_eqn_name(eqn)} but no value was provided.")
                outvals = fixed[_eqn_name(eqn)]
            else:
                invals = safe_map(read, eqn.invars)
                subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
                outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
            # Save to context
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            safe_map(write, eqn.outvars, outvals)
        # Return output
        out = safe_map(read, jaxpr.outvars)
        return out[0] if len(out) == 1 else out

    # Forward pass returning intermediates
    def intermediates(params, **fixed):
        """Evaluate model and return intermediate values with params and (optionally) fixed values in the structure returned by `zeros`/`normal` and `fixed`, respectively."""
        if unravel:
            params = unravel(params)
        env = {}

        def read(v):
            return v.val if type(v) is core.Literal else env[v]

        def write(v, val):
            env[v] = val

        # Bind args and consts to environment
        safe_map(write, jaxpr.constvars, jaxpr.consts)

        # Loop through equations and evaluate
        intermediate_outs = {}
        for eqn in jaxpr.jaxpr.eqns:
            # Load parameter or bind primitive
            if eqn.primitive is param_p:
                if _eqn_name(eqn) not in params:
                    raise ValueError(f"Model has parameter {_eqn_name(eqn)} but no value was provided.")
                outvals = params[_eqn_name(eqn)]
            elif eqn.primitive is fixed_p:
                if _eqn_name(eqn) not in fixed:
                    raise ValueError(f"Model has fixed parameter {_eqn_name(eqn)} but no value was provided.")
                outvals = fixed[_eqn_name(eqn)]
            else:
                invals = safe_map(read, eqn.invars)
                subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
                outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
            # Save to context
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            safe_map(write, eqn.outvars, outvals)
            # If intermediate, save to output dict
            if eqn.primitive is intermediate_p:
                intermediate_outs[_eqn_name(eqn)] = outvals[0] if len(outvals) == 1 else outvals

        return intermediate_outs

    forward.shapes = shapes
    forward.normal = normal
    forward.zeros = zeros
    forward.intermediates = intermediates
    forward.fixed = fixed
    forward.unravel = unravel

    return forward
