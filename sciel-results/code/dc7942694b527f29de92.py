"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 120.0
_START = _time.time()
_MARGIN = 5.0  # seconds to stay under the hard limit

# Lightweight bookkeeping: approximate machine load so we can balance.
# We do not receive completion events, so we accumulate a running estimate
# of how much work has been *assigned* to each machine. This is a proxy
# that helps spread load across machines within a single run.
_MACHINE_LOAD = {}
_LAST_TIME = -1.0

def _machine_load(m):
    return _MACHINE_LOAD.get(m, 0.0)

def _bump_machine_load(m, d):
    _MACHINE_LOAD[m] = _MACHINE_LOAD.get(m, 0.0) + d


def dispatch(ready_operations, time):
    global _LAST_TIME
    
    # Reset state if time goes backwards or resets to 0 (new instance)
    if time < _LAST_TIME or time == 0:
        _MACHINE_LOAD.clear()
    _LAST_TIME = time

    # Deadline guard: if we are about to exceed the hard limit, fall back
    # to the simple baseline (argmax remaining_work) which is O(n) and safe.
    if _time.time() - _START > _DEADLINE - _MARGIN:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # ---- Build per-operation features ---------------------------------
    # job urgency: total remaining work for the job (higher = more urgent)
    # intensity:   work per remaining op (normalizes jobs of different size)
    # own_proc:    minimum processing time over its machine options
    #              (shorter ops first reduces machine idle / blocking)
    # flex:        range of processing times (max - min)
    #              (lower flex = more constrained, should be scheduled earlier)
    urg = []
    inten = []
    own = []
    flex = []
    for op in ready_operations:
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1)) or 1.0
        mo = op.get("machine_options", {})
        # minimum duration among eligible machines
        if mo:
            durs = [float(v) for v in mo.values()]
            own_proc = min(durs)
            max_proc = max(durs)
            flex.append(max_proc - own_proc)
        else:
            own_proc = rw / ro
            flex.append(0.0)
        urg.append(rw)
        inten.append(rw / ro)
        own.append(own_proc)

    urg = np.asarray(urg, dtype=float)
    inten = np.asarray(inten, dtype=float)
    own = np.asarray(own, dtype=float)
    flex = np.asarray(flex, dtype=float)

    # Normalise into [0,1] to make weights comparable.
    def _norm(x):
        rng = x.max() - x.min()
        if rng <= 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / rng

    u = _norm(urg)
    i = _norm(inten)
    o = _norm(own)  # we actually prefer SMALL own_proc, so invert below
    f = _norm(flex)  # we prefer LOW flex (constrained), so invert below

    # ---- Choose the operation -----------------------------------------
    # We use a smooth surrogate for the max-like urgency objective via a
    # soft-max over the normalized features. This avoids the kink at the
    # argmax and may generalize better across instances.
    #
    # We also increase the weight on SPT (short processing time) slightly,
    # as shorter operations first tend to reduce blocking and machine idle
    # time in flexible job-shop settings.
    #
    # Weights: Urgency 0.30, Intensity 0.10, SPT 0.50, Constraint 0.10
    w_u, w_i, w_o, w_f = 0.30, 0.10, 0.50, 0.10
    
    # Compute a smooth combination using a log-sum-exp-like soft-max
    # over the weighted, normalized features. We invert o and f so that
    # higher values mean "prefer this operation".
    soft_u = w_u * u
    soft_i = w_i * i
    soft_o = w_o * (1.0 - o)
    soft_f = w_f * (1.0 - f)
    
    # Sum the soft features and use a soft-max to get a smooth score.
    # The temperature parameter controls how close to the true max we are.
    # A small temperature gives a smoother, more robust score.
    temperature = 0.1
    total = soft_u + soft_i + soft_o + soft_f
    op_score = temperature * np.log(np.sum(np.exp(total / temperature), axis=0))
    
    # If all operations have the same total, fall back to a simple argmax
    # to avoid numerical issues.
    if np.std(op_score) < 1e-9:
        op_score = total

    best_idx = int(np.argmax(op_score))
    # If op_score was all equal, use total as tiebreaker
    if np.std(op_score) < 1e-9:
        best_idx = int(np.argmax(total))

    # ---- Choose the machine (load balancing) --------------------------
    chosen_op = ready_operations[best_idx]
    mo = chosen_op.get("machine_options", {})
    if mo:
        # Pick the eligible machine that minimizes a smooth combination of
        # current load and processing duration. We use a soft-min to avoid
        # abrupt switches and encourage more balanced load distribution.
        best_m = None
        best_key = None
        # Temperature for soft-min: smaller = closer to true min
        temp_m = 0.05
        keys = []
        for m, dur in mo.items():
            key = _machine_load(m) + float(dur)
            keys.append((m, key, float(dur)))
        
        if keys:
            # Use soft-min: -temp * log(sum(exp(-key/temp)))
            # This gives a smooth approximation to min(key)
            neg_keys = np.array([-k[1] / temp_m for k in keys])
            log_sum_exp = temp_m * np.log(np.sum(np.exp(neg_keys)))
            # The machine with the lowest key gets the highest weight
            # We pick the argmin directly but could also use a weighted approach
            best_m = min(keys, key=lambda x: x[1])[0]
            best_d = [k[2] for k in keys if k[0] == best_m][0]
            _bump_machine_load(best_m, best_d)
            return (best_idx, best_m)

    # Fallback: update load for the (unknown) machine is impossible; just
    # return the index.
    return best_idx
# EVOLVE-BLOCK-END
