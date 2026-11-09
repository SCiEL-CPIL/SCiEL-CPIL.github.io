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
    return (min(vs), max(vs))


def dispatch(ready_operations, time):
    if _elapsed() > _LIMIT:
        return 0
    n_ops = len(ready_operations)
    if n_ops <= 1:
        return 0

    def _priority(op, idx):
        rem = op["remaining_work"]
        mn, mx = _duration_stats(op["machine_options"])
        return (rem, mx, -mn, -idx)

    ranked = [( _priority(op, i), i ) for i, op in enumerate(ready_operations)]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]
# EVOLVE-BLOCK-END
