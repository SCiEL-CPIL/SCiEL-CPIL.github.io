import time as time_module
import math

# ------------------------------------------------------------------
#  Open-Shop dispatch rule (refined within the multi-criterion
#  heuristic family).
#
#  Changes from the seed:
#   * Tighter LSE temperature (2.0 vs 5.0) so the smooth makespan
#     surrogate tracks the true max more closely, reducing the
#     "blurring" that lets non-critical ops compete unfairly.
#   * Added a per-job critical-path weight: remaining_ops is used
#     as a proxy for the number of operations still in the job's
#     chain; jobs with more remaining ops are on longer critical
#     paths and should be prioritised.
#   * Machine-load penalty uses the *max* load among the op's
#     candidate machines (worst-case congestion) rather than the
#     min, giving a stronger incentive to spread load.
#   * Beam width widened to 6 and top-K to 6; the look-ahead
#     simulation now also subtracts the chosen op's remaining_work
#     from the pool before computing the surrogate, giving a more
#     accurate post-selection makespan estimate.
#   * Removed numpy dependency (math.log is sufficient and faster
#     for the small vector sizes here), reducing import overhead.
#   * All weights re-tuned: W_MWR up, W_MK up (makespan dominates),
#     W_SPT down, W_MACH up, W_URG (now driven by remaining_ops)
#     added at 0.3.
# ------------------------------------------------------------------

_DEADLINE = 150.0
_START = time_module.time()
_MARGIN = 2.0

# Tuned weights
W_MWR  = 1.0     # max remaining work (higher = more remaining work preferred)
W_SPT  = 0.10    # short-processing-time bonus
W_MACH = 0.10    # machine-load balance penalty
W_URG  = 0.30    # urgency: more remaining ops -> higher priority
W_MK   = 0.8     # makespan surrogate penalty (lower is better)

_LSE_TEMP = 2.0   # sharper log-sum-exp
_BEAM    = 6
_TOP_K   = 6


def _elapsed():
    return time_module.time() - _START


def _remaining():
    return _DEADLINE - _MARGIN - _elapsed()


def _lse2(a, b, temp):
    """Smooth max of two values via log-sum-exp."""
    m = a if a > b else b
    if temp <= 0:
        return m
    return m + temp * math.log(math.exp((a - m) / temp) + math.exp((b - m) / temp))


def _count_loads(ops):
    loads = {}
    for op in ops:
        for m in op.get("machine_options", {}):
            loads[m] = loads.get(m, 0) + 1
    return loads


def _est_finish(ops, t):
    est = {}
    for op in ops:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        md = min(mo.values())
        for m in mo:
            est[m] = est.get(m, 0.0) + md
    for m in est:
        est[m] += t
    return est


def _score(op, loads, t, mfin):
    mo = op.get("machine_options", {})
    rw = op.get("remaining_work", 0.0)
    rem_ops = op.get("remaining_ops", [])
    n_rem = len(rem_ops) if isinstance(rem_ops, (list, tuple)) else 0

    durs = list(mo.values()) if mo else [0.0]
    min_dur = min(durs) if durs else 0.0

    # Worst-case machine congestion among candidate machines
    if mo:
        max_load = max(loads.get(m, 0) for m in mo)
    else:
        max_load = 0

    # Estimated completion on the best machine
    if mfin is not None and mo:
        best_comp = min(mfin.get(m, t) + mo[m] for m in mo)
    else:
        best_comp = t + min_dur

    # Smooth makespan surrogate
    mk = _lse2(best_comp, rw, _LSE_TEMP)

    s = (W_MWR * rw
         + W_SPT * (1.0 / (min_dur + 1.0))
         - W_MACH * max_load
         + W_URG * (n_rem / (n_rem + 1.0))
         - W_MK * mk)
    return s


def _lookahead_score(ops, idx, t, loads, mfin):
    """
    1-step look-ahead: tentatively assign op[idx], recompute a
    makespan surrogate over the *remaining* ready ops, and return
    the composite score.
    """
    op = ops[idx]
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0

    # Pick the machine with the earliest completion
    best_m = min(mo, key=lambda m: (mfin.get(m, t) if mfin else t) + mo[m])
    comp = (mfin.get(best_m, t) if mfin else t) + mo[best_m]

    # Update loads: remove this op from all its machines, add to best_m
    new_loads = dict(loads)
    for m in mo:
        if m in new_loads:
            new_loads[m] -= 1
    new_loads[best_m] = new_loads.get(best_m, 0) + 1

    # Update est finish for best_m
    new_mfin = dict(mfin) if mfin else {}
    if best_m in new_mfin:
        new_mfin[best_m] = comp
    else:
        new_mfin[best_m] = comp

    # Compute the makespan surrogate over the remaining ops
    # and the just-completed op's finish time.
    max_rw = 0.0
    for j, other in enumerate(ops):
        if j != idx:
            rw2 = other.get("remaining_work", 0.0)
            if rw2 > max_rw:
                max_rw = rw2

    # Smooth max over {comp, max_rw}
    mk_after = _lse2(comp, max_rw, _LSE_TEMP)

    # Also score the remaining ops to get a "state quality" signal
    state_score = 0.0
    for j, other in enumerate(ops):
        if j == idx:
            continue
        mo2 = other.get("machine_options", {})
        durs2 = list(mo2.values()) if mo2 else [0.0]
        md2 = min(durs2) if durs2 else 0.0
        if mo2:
            ml2 = max(new_loads.get(m, 0) for m in mo2)
            bc2 = min(new_mfin.get(m, t) + mo2[m] for m in mo2)
        else:
            ml2 = 0
            bc2 = t + md2
        rw3 = other.get("remaining_work", 0.0)
        rem2 = other.get("remaining_ops", [])
        nr2 = len(rem2) if isinstance(rem2, (list, tuple)) else 0
        mk2 = _lse2(bc2, rw3, _LSE_TEMP)
        state_score += (W_MWR * rw3
                        + W_SPT * (1.0 / (md2 + 1.0))
                        - W_MACH * ml2
                        + W_URG * (nr2 / (nr2 + 1.0))
                        - W_MK * mk2)

    # Composite: lower makespan surrogate + state quality
    return -0.5 * mk_after + 0.5 * state_score


def _fast(ops, t):
    loads = _count_loads(ops)
    mfin = _est_finish(ops, t)
    best_i = 0
    best_s = float('-inf')
    for i, op in enumerate(ops):
        s = _score(op, loads, t, mfin)
        if s > best_s:
            best_s = s
            best_i = i
    return best_i


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if _remaining() < 0.3:
        return _fast(ready_operations, time)

    n = len(ready_operations)
    loads = _count_loads(ready_operations)
    mfin = _est_finish(ready_operations, time)

    # Score all
    scored = []
    for i, op in enumerate(ready_operations):
        scored.append((_score(op, loads, time, mfin), i))
    scored.sort(key=lambda x: x[0], reverse=True)

    # If very few ops, no need for look-ahead
    if n <= 2:
        return scored[0][1]

    # Beam search over top-K with 1-step look-ahead
    k = min(_TOP_K, n)
    best_idx = scored[0][1]
    best_combined = scored[0][0] - 0.1 * (mfin.get(min(ready_operations[best_idx].get("machine_options", {0: 1}), default=0), time) + 1.0, time)[0] if False else float('-inf')
    # Recompute properly:
    best_combined = float('-inf')

    for r in range(k):
        base_s, idx = scored[r]
        la = _lookahead_score(ready_operations, idx, time, loads, mfin)
        combined = base_s + 0.3 * la
        if combined > best_combined:
            best_combined = combined
            best_idx = idx
        if _remaining() < 1.0:
            break

    return best_idx
