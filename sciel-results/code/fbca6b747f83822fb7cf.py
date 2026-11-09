"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _clock
import math

_T0 = _clock.time()
_LIMIT = 135.0


def _elapsed():
    return _clock.time() - _T0


def _duration_stats(opts):
    if not opts:
        return (0.0, 0.0)
    vs = list(opts.values())
    if len(vs) == 1:
        return (vs[0], vs[0])
    # Smooth surrogate: log-sum-exp for max, negative log-sum-exp for min
    # Use a moderate temperature to smooth the nonsmooth min/max
    import math as _m
    M = max(vs)
    m = min(vs)
    if M == m:
        return (M, M)
    # Soft-max: M + log(sum(exp(v - M)))
    soft_max = M + _m.log(sum(_m.exp(v - M) for v in vs))
    # Soft-min: m - log(sum(exp(m - v)))
    soft_min = m - _m.log(sum(_m.exp(m - v) for v in vs))
    return (soft_min, soft_max)


def dispatch(ready_operations, time):
    if _elapsed() > _LIMIT:
        return 0
    n_ops = len(ready_operations)
    if n_ops <= 1:
        return 0

    def _priority(op, idx):
        rem = op["remaining_work"]
        mn, mx = _duration_stats(op["machine_options"])
        rem_ops = op.get("remaining_ops", 0)
        # Load-aware: prefer ops on less-congested machines (shorter durations)
        # Combine remaining work (urgency) with machine load (congestion)
        # Higher remaining_work = more urgent; lower mx = less congested
        # Use a weighted blend: urgency dominates, but machine load breaks ties
        priority = rem - 0.5 * mx + 0.3 * mn - 0.1 * rem_ops
        return (priority, -idx)

    ranked = [( _priority(op, i), i ) for i, op in enumerate(ready_operations)]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
