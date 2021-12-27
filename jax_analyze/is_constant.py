import jax
from jax import core
from jax._src.util import safe_map


def is_provably_constant(closed_jaxpr: jax.core.ClosedJaxpr) -> bool:
    return _is_const(closed_jaxpr.jaxpr)


def _is_const(jaxpr):
    env = {}

    def read(var):
        if type(var) is core.Literal:
            return True
        return env[var]

    def write(var, value):
        if type(var) is core.Literal:
            assert value
            return
        env[var] = value

    safe_map(write, jaxpr.invars, [False for _ in jaxpr.invars])
    safe_map(write, jaxpr.constvars, [True for _ in jaxpr.constvars])

    for eqn in jaxpr.eqns:
        if eqn.primitive.multiple_results:
            raise NotImplementedError()
        invars_const = safe_map(read, eqn.invars)
        all_invars_const = jax.numpy.all(jax.numpy.stack(invars_const))
        write(eqn.outvars[0], all_invars_const)
        pass

    is_const = jax.numpy.all(jax.numpy.stack(safe_map(read, jaxpr.outvars)))
    return is_const
