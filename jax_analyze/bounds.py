import jax
import jax.numpy as jnp
from jax import core, lax
from jax._src.util import safe_map
from jax._src.ad_util import add_any_p

"""
bound_fn: function where each element is substituted by a 2-tuple
"""


@jax.tree_util.register_pytree_node_class
class IntervalBound:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def tree_flatten(self):
        return (self.lower, self.upper), ()

    @classmethod
    def tree_unflatten(cls, meta, data):
        return cls(*data)

    def __repr__(self):
        return f"IB({self.lower}, {self.upper})"


#

def _add_bounds(eqn: core.JaxprEqn):
    def bounds_fn(lhs: IntervalBound, rhs: IntervalBound) -> IntervalBound:
        return IntervalBound(lhs.lower + rhs.lower, lhs.upper + rhs.upper)

    return bounds_fn


def _sub_bounds(eqn: core.JaxprEqn):
    def bounds_fn(lhs: IntervalBound, rhs: IntervalBound) -> IntervalBound:
        return IntervalBound(lhs.lower - rhs.upper, lhs.upper - rhs.lower)

    return bounds_fn


def _neg_bounds(eqn: core.JaxprEqn):
    def bounds_fn(x: IntervalBound) -> IntervalBound:
        return IntervalBound(- x.upper, - x.lower)

    return bounds_fn


def _monotone_fn_bounds(eqn: core.JaxprEqn):
    def bounds_fn(x: IntervalBound):
        return IntervalBound(eqn.primitive.bind(x.lower), eqn.primitive.bind(x.upper))

    return bounds_fn


def _mul_bounds(eqn: core.JaxprEqn):
    if eqn.invars[0] == eqn.invars[1]:
        def bounds_fn_eq(lhs: IntervalBound, rhs: IntervalBound):
            lwb = (lhs.lower * rhs.lower > 0.0) * lax.min(lhs.lower ** 2.0, lhs.upper ** 2.0)
            upb = jnp.maximum(lhs.lower * rhs.lower, lhs.upper * rhs.upper)
            return IntervalBound(lwb, upb)

        return bounds_fn_eq
        pass

    def bounds_fn(lhs: IntervalBound, rhs: IntervalBound):
        a = [lhs.lower * rhs.lower, lhs.lower * rhs.upper, lhs.upper * rhs.lower, lhs.upper * rhs.upper]
        lwb = jnp.minimum(jnp.minimum(a[0], a[1]), jnp.minimum(a[2], a[3]))
        upb = jnp.maximum(jnp.maximum(a[0], a[1]), jnp.maximum(a[2], a[3]))
        return IntervalBound(lwb, upb)

    return bounds_fn


def _reduce_sum_p(eqn: core.JaxprEqn):
    def bounds_fn(x: IntervalBound) -> IntervalBound:
        return IntervalBound(lax.reduce_sum_p.bind(x.lower, **eqn.params), lax.reduce_sum_p.bind(x.upper, **eqn.params))

    return bounds_fn


def _lt_bounds(eqn: core.JaxprEqn):
    def bounds_fn(lhs: IntervalBound, rhs: IntervalBound):
        a = [lhs.lower < rhs.lower, lhs.lower < rhs.upper, lhs.upper < rhs.lower, lhs.upper < rhs.upper]
        lwb = jnp.minimum(jnp.minimum(a[0], a[1]), jnp.minimum(a[2], a[3]))
        upb = jnp.maximum(jnp.maximum(a[0], a[1]), jnp.maximum(a[2], a[3]))
        return IntervalBound(lwb, upb)

    return bounds_fn


def _select_bounds(eqn: core.JaxprEqn):
    def bounds_fn(a: IntervalBound, b: IntervalBound, c: IntervalBound) -> IntervalBound:
        lwb = jnp.minimum(lax.select(a.lower, b.lower, c.lower), lax.select(a.upper, b.lower, c.lower))
        upb = jnp.maximum(lax.select(a.lower, b.upper, c.upper), lax.select(a.upper, b.upper, c.upper))
        return IntervalBound(lwb, upb)

    return bounds_fn


def _reshape_bounds(eqn: core.JaxprEqn):
    def bounds_fn(x: IntervalBound):
        lwb = eqn.primitive.bind(x.lower, **eqn.params)
        upb = eqn.primitive.bind(x.upper, **eqn.params)
        return IntervalBound(lwb, upb)

    return bounds_fn


def _gather_bounds(eqn: core.JaxprEqn):
    def bounds_fn(a: IntervalBound, b: IntervalBound):
        lwb = jnp.minimum(eqn.primitive.bind(a.lower, b.lower, **eqn.params),
                          eqn.primitive.bind(a.lower, b.upper, **eqn.params))
        upb = jnp.maximum(eqn.primitive.bind(a.upper, b.lower, **eqn.params),
                          eqn.primitive.bind(a.upper, b.lower, **eqn.params))
        return IntervalBound(lwb, upb)

    return bounds_fn


_bound_from_primitive = {
    lax.add_p: _add_bounds,
    add_any_p: _add_bounds,
    lax.exp_p: _monotone_fn_bounds,
    lax.log_p: _monotone_fn_bounds,
    lax.mul_p: _mul_bounds,
    lax.reduce_sum_p: _reduce_sum_p,
    lax.sub_p: _sub_bounds,
    lax.neg_p: _neg_bounds,
    lax.lt_p: _lt_bounds,
    lax.select_p: _select_bounds,
    lax.gather_p: _gather_bounds,
    # RESHAPE
    lax.broadcast_in_dim_p: _reshape_bounds,
    lax.reshape_p: _reshape_bounds
}


def bounds_fn_from_eqn(eqn: core.JaxprEqn):
    bound_fn = _bound_from_primitive[eqn.primitive](eqn)
    return bound_fn


#
def eval_bounds_from_jaxpr(jaxpr: core.Jaxpr, consts, *args):
    env = {}

    def read(var):
        if type(var) is core.Literal:
            return IntervalBound(var.val, var.val)
        return env[var]

    def write(var, value: IntervalBound):
        env[var] = value

    safe_map(write, jaxpr.invars, args)

    for eqn in jaxpr.eqns:
        inbounds = safe_map(read, eqn.invars)
        outbounds = bounds_fn_from_eqn(eqn)(*inbounds)
        assert len(eqn.outvars) == 1
        write(eqn.outvars[0], outbounds)
        pass
    return safe_map(read, jaxpr.outvars)


def bounds(fun):
    def wrapped(*args):
        placeholder_args = jax.tree_util.tree_map(lambda x: x.lower if isinstance(x, IntervalBound) else x,
                                                  args,
                                                  is_leaf=lambda x: isinstance(x, IntervalBound))
        closed_jaxpr = jax.make_jaxpr(fun)(*placeholder_args)
        output_shapes = jax.eval_shape(fun, *placeholder_args)

        #
        args_lwb = jax.tree_map(lambda x: x.lower if isinstance(x, IntervalBound) else x, args,
                                is_leaf=lambda x: isinstance(x, IntervalBound))
        args_upb = jax.tree_map(lambda x: x.upper if isinstance(x, IntervalBound) else x, args,
                                is_leaf=lambda x: isinstance(x, IntervalBound))
        args_ = jax.tree_multimap(lambda x, y: IntervalBound(x, y), args_lwb, args_upb)

        args_leaves, args_pytree_def = jax.tree_flatten(args_, is_leaf=lambda x: isinstance(x, IntervalBound))
        out_leaves = eval_bounds_from_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args_leaves)
        _, out_pytree_def = jax.tree_flatten(output_shapes)
        out = jax.tree_unflatten(out_pytree_def, out_leaves)
        return out

    return wrapped


def lower_bound(fun):
    def wrapped(*args):
        placeholder_args = jax.tree_util.tree_map(lambda x: x.lower if isinstance(x, IntervalBound) else x,
                                                  args,
                                                  is_leaf=lambda x: isinstance(x, IntervalBound))
        closed_jaxpr = jax.make_jaxpr(fun)(*placeholder_args)
        output_shapes = jax.eval_shape(fun, *placeholder_args)

        #
        args_lwb = jax.tree_map(lambda x: x.lower if isinstance(x, IntervalBound) else x, args,
                                is_leaf=lambda x: isinstance(x, IntervalBound))
        args_upb = jax.tree_map(lambda x: x.upper if isinstance(x, IntervalBound) else x, args,
                                is_leaf=lambda x: isinstance(x, IntervalBound))
        args_ = jax.tree_multimap(lambda x, y: IntervalBound(x, y), args_lwb, args_upb)

        args_leaves, args_pytree_def = jax.tree_flatten(args_, is_leaf=lambda x: isinstance(x, IntervalBound))
        out_leaves = eval_bounds_from_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args_leaves)
        out_leaves = [out_i.lower for out_i in out_leaves]
        _, out_pytree_def = jax.tree_flatten(output_shapes)
        out = jax.tree_unflatten(out_pytree_def, out_leaves)
        return out

    return wrapped


def upper_bound(fun):
    def wrapped(*args):
        placeholder_args = jax.tree_util.tree_map(lambda x: x.lower if isinstance(x, IntervalBound) else x,
                                                  args,
                                                  is_leaf=lambda x: isinstance(x, IntervalBound))
        closed_jaxpr = jax.make_jaxpr(fun)(*placeholder_args)
        output_shapes = jax.eval_shape(fun, *placeholder_args)

        #
        args_lwb = jax.tree_map(lambda x: x.lower if isinstance(x, IntervalBound) else x, args,
                                is_leaf=lambda x: isinstance(x, IntervalBound))
        args_upb = jax.tree_map(lambda x: x.upper if isinstance(x, IntervalBound) else x, args,
                                is_leaf=lambda x: isinstance(x, IntervalBound))
        args_ = jax.tree_multimap(lambda x, y: IntervalBound(x, y), args_lwb, args_upb)

        args_leaves, args_pytree_def = jax.tree_flatten(args_, is_leaf=lambda x: isinstance(x, IntervalBound))
        out_leaves = eval_bounds_from_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args_leaves)
        out_leaves = [out_i.upper for out_i in out_leaves]
        _, out_pytree_def = jax.tree_flatten(output_shapes)
        out = jax.tree_unflatten(out_pytree_def, out_leaves)
        return out

    return wrapped
