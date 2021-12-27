import jax
import jax.numpy as jnp
import jax_analyze
from jax_analyze import IntervalBound


def fn(x, y):
    return jnp.exp(x[0]) + 0.5 * y['a']


fn_lwb = jax_analyze.lower_bound(fn)
input_bounds = [IntervalBound(jnp.zeros(2, ), jnp.ones(2, )), {'a': IntervalBound(jnp.array(0.0), jnp.array(1.0))}]
print(fn_lwb(*input_bounds))
