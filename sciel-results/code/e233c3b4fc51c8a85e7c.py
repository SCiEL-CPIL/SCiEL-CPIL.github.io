import numpy as np
import time as time_module

# ------------------------------------------------------------------
#  Open-Shop dispatch rule (REFINED – same design family as parent).
#
#  Core idea (critical-path priority + smooth makespan surrogate):
#   * The open-shop makespan is governed by the longest "tail" of
#     remaining work.  We approximate it with a log-sum-exp (soft-max)
#     over the *estimated completion time* of every ready operation,
#     where an op's completion = (machine busy lower-bound on its best
#     machine) + its processing duration.  Minimising this surrogate
#     directly attacks the makespan.
#   * Machine busy lower-bound: for each machine, sum of the *minimum*
#     duration over all ready ops that can use it, plus the current
#     time.  This is a valid lower bound on when the machine can finish
#     its ready workload, so the completion estimates are optimistic but
#     consistent (no double counting of the same op on the same machine).
#   * Secondary (critical-path) signal: the job with the largest total
#     remaining work is the longest-tail job; we prefer its ready ops.
#   * A 1-step look-ahead refines the choice among the top-K candidates
#     by re-estimating the surrogate after committing to a candidate.
#
#  All timing is guarded by an explicit wall-clock deadline with a
#  safety margin; a fast O(n) path is used when time is critical.
# ------------------------------------------------------------------

_DEADLINE = 148.0
_START_TIME = time_module.time()
_TIME_MARGIN = 3.0

# Weights for the composite per-op score (primary = makespan surrogate).
W_MK    = 1.00    # soft-max makespan surrogate (dominant)
W_TAIL  = 0.35    # job critical-path (total remaining work) urgency
W_BAL   = 0.08    # machine load balance nudge

_LSE_TEMP   = 4.0    # soft-max temperature (lower = sharper max)
_TOP_K      = 6      # beam width for look-ahead refinement


def _elapsed():
    return time_module.time() - _START_TIME


def _remaining_time():
    return _DEADLINE - _TIME_MARGIN - _elapsed()


def _soft_max(vals, temp):
    """Smooth max via log-sum-exp (stable)."""
    if not vals:
        return 0.0
    m_val = max(vals)
    if m_val == float('inf'):
        return m_val
    if temp <= 0:
        return m_val
    s = 0.0
    for v in vals:
        s += np.exp((v - m_val) / temp)
    return m_val + temp * np.log(s)


def _build_context(ready_operations, time):
    """
    Pre-compute:
      * job_tail[job]  : total remaining work of the job (critical path)
      * machine_lb[m]  : lower bound on machine m's busy interval end
                         (time + sum of min durations of ready ops on m)
      * op_min_dur[i]  : min duration over op i's machine options
    """
    job_tail = {}
    machine_lb = {}
    op_min_dur = []

    for op in ready_operations:
        job = op.get("job")
        rw = op.get("remaining_work", 0.0)
        if job is not None:
            job_tail[job] = job_tail.get(job, 0.0) + rw

        mo = op.get("machine_options", {})
        if not mo:
            op_min_dur.append(0.0)
            continue
        mn = min(mo.values())
        op_min_dur.append(mn)
        for m in mo:
            # lower bound: each ready op contributes at least its min dur
            machine_lb[m] = machine_lb.get(m, 0.0) + mn

    for m in machine_lb:
        machine_lb[m] += time

    return job_tail, machine_lb, op_min_dur


def _best_completion(op, machine_lb, time, op_min_dur):
    """
    Earliest estimated completion for this op across its machine options.
    Uses the machine lower-bound (busy interval) + the op's duration on that
    machine.  Returns (machine, completion) or (None, time).
    """
    mo = op.get("machine_options", {})
    if not mo:
        return None, time
    best_m = None
    best_t = float('inf')
    for m, d in mo.items():
        t = machine_lb.get(m, time) + d
        if t < best_t:
            best_t = t
            best_m = m
    return best_m, best_t


def _makespan_surrogate(ready_operations, machine_lb, time, op_min_dur):
    """Soft-max over estimated completion times of all ready ops."""
    vals = []
    for i, op in enumerate(ready_operations):
        _, t = _best_completion(op, machine_lb, time, op_min_dur)
        vals.append(t)
    return _soft_max(vals, _LSE_TEMP)


def _score_op(op, i, job_tail, machine_lb, time, op_min_dur):
    """
    Composite per-op score.  Higher = more preferred.

    Primary: minimise the makespan surrogate contribution of this op
             (its estimated completion time).
    Secondary: job critical-path urgency (larger total remaining work).
    Tertiary: load balance (prefer less-loaded best machine).
    """
    mo = op.get("machine_options", {})
    best_m, best_comp = _best_completion(op, machine_lb, time, op_min_dur)
    rw = op.get("remaining_work", 0.0)

    # Makespan surrogate term: this op's completion time (lower better).
    mk_term = best_comp

    # Critical-path term: job's total remaining work (larger better).
    job = op.get("job")
    tail = job_tail.get(job, rw) if job is not None else rw

    # Load-balance term: number of ready ops that can use the best machine.
    if best_m is not None:
        bal = machine_lb.get(best_m, time) - time  # proxy for load (min-dur sum)
    else:
        bal = 0.0

    # Normalise to a comparable scale (~ time).
    scale = max(time, 1.0)
    score = (W_MK   * (-mk_term / scale)
             + W_TAIL * (tail / scale)
             - W_BAL  * (bal / scale))
    return score


def _fast_dispatch(ready_operations, time):
    """O(n) heuristic when time is critical."""
    job_tail, machine_lb, op_min_dur = _build_context(ready_operations, time)
    best_idx, best_s = 0, float('-inf')
    for i, op in enumerate(ready_operations):
        s = _score_op(op, i, job_tail, machine_lb, time, op_min_dur)
        if s > best_s:
            best_s, best_idx = s, i
    return best_idx


def _commit(ready_operations, idx, machine_lb, time, op_min_dur):
    """
    Return an updated machine lower-bound dict after committing to
    ready_operations[idx] on its best machine.  The op's contribution is
    removed from the lower bounds of the machines it could use (its min
    duration) and replaced by its actual duration on the chosen machine.
    """
    op = ready_operations[idx]
    mo = op.get("machine_options", {})
    new_lb = dict(machine_lb)
    if not mo:
        return new_lb
    mn = op_min_dur[idx]
    for m in mo:
        if m in new_lb:
            new_lb[m] = max(time, new_lb[m] - mn)
    best_m, _ = _best_completion(op, machine_lb, time, op_min_dur)
    if best_m is not None:
        new_lb[best_m] = max(time, new_lb.get(best_m, time)) + mo[best_m]
    return new_lb


def _lookahead_1(ready_operations, idx, machine_lb, time, op_min_dur):
    """Makespan surrogate after committing to ready_operations[idx]."""
    new_lb = _commit(ready_operations, idx, machine_lb, time, op_min_dur)
    return _makespan_surrogate(ready_operations, new_lb, time, op_min_dur)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Deadline guard: fall back to fast path
    if _remaining_time() < 1.0:
        return _fast_dispatch(ready_operations, time)

    # Build shared context
    job_tail, machine_lb, op_min_dur = _build_context(ready_operations, time)

    # Score all ops
    scores = []
    for i, op in enumerate(ready_operations):
        s = _score_op(op, i, job_tail, machine_lb, time, op_min_dur)
        scores.append((s, i))

    scores.sort(key=lambda x: x[0], reverse=True)

    if n <= 2:
        return scores[0][1]

    # --- Look-ahead refinement over top-K candidates ---
    # The base score already encodes the makespan surrogate; the look-ahead
    # re-estimates the surrogate after committing, giving a sharper signal
    # for breaking near-ties among the top candidates.
    top_k = min(_TOP_K, n)

    best_idx = scores[0][1]
    best_la = _lookahead_1(ready_operations, scores[0][1], machine_lb, time, op_min_dur)
    best_combined = scores[0][0] - 0.5 * best_la / max(time, 1.0)

    for k in range(1, top_k):
        idx = scores[k][1]
        base = scores[k][0]
        la = _lookahead_1(ready_operations, idx, machine_lb, time, op_min_dur)
        combined = base - 0.5 * la / max(time, 1.0)

        if combined > best_combined:
            best_combined = combined
            best_idx = idx
            best_la = la

        if _remaining_time() < 1.5:
            break

    return best_idx
