"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Persistent state across dispatch calls within a single simulation run.
# We reset when we detect a new simulation (via a sentinel).
_state = {
    "machine_load": {},   # machine -> cumulative duration assigned to that machine
    "job_load": {},       # job -> cumulative duration assigned to that job
    "seen_ids": set(),    # to detect a fresh simulation
    "op_counter": 0,
}


def _reset_state(ready_operations):
    _state["machine_load"] = {}
    _state["job_load"] = {}
    _state["seen_ids"] = set()
    _state["op_counter"] = 0


def dispatch(ready_operations, time):
    # Hard deadline guard: if we've spent too long, fall back to a cheap rule.
    # (The dispatch itself is fast, but guard defensively.)
    n = len(ready_operations)
    if n == 0:
        return 0

    # Detect a new simulation: if none of the ready ops' identities were seen
    # before AND we already have non-trivial state, assume a fresh run.
    # We key on (job, op) pairs; a fresh sim will have a disjoint set.
    ids = set()
    for op in ready_operations:
        ids.add((op.get("job"), op.get("op")))
    if _state["seen_ids"] and not (ids & _state["seen_ids"]):
        _reset_state(ready_operations)
    _state["seen_ids"] |= ids

    ml = _state["machine_load"]
    jl = _state["job_load"]

    # Fast path: single option per op -> simple scoring.
    best_idx = 0
    best_score = -1e300
    best_assign = None  # (machine, duration) to commit

    for i, op in enumerate(ready_operations):
        job = op.get("job")
        opts = op.get("machine_options")
        if not opts:
            continue
        # Choose the machine for this job that minimizes resulting machine load
        # (balance), breaking ties by shorter duration.
        chosen_machine = None
        chosen_dur = None
        min_key = None
        for m, dur in opts.items():
            cur = ml.get(m, 0.0)
            key = (cur + dur, dur)
            if min_key is None or key < min_key:
                min_key = key
                chosen_machine = m
                chosen_dur = dur
        if chosen_machine is None:
            continue

        rem_work = float(op.get("remaining_work", 0.0) or 0.0)
        rem_ops = op.get("remaining_ops", 0) or 0
        j_cur = jl.get(job, 0.0)

        # Score: prioritize jobs with more remaining work (urgency),
        # then prefer assignments that keep machine load low (balance).
        # We combine: urgency dominates, balance is a secondary tie-breaker
        # scaled to be meaningful but not overwhelming.
        machine_after = ml.get(chosen_machine, 0.0) + chosen_dur
        score = rem_work - 0.05 * machine_after - 0.02 * (j_cur + chosen_dur)
        if score > best_score:
            best_score = score
            best_idx = i
            best_assign = (chosen_machine, chosen_dur)

    # Commit the chosen assignment to persistent state.
    if best_assign is not None:
        m, d = best_assign
        ml[m] = ml.get(m, 0.0) + d
        job = ready_operations[best_idx].get("job")
        jl[job] = jl.get(job, 0.0) + d

    return int(best_idx)
# EVOLVE-BLOCK-END
