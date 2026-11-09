"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# --- deadline guard ---
_START = None
_DEADLINE = 150.0
_MARGIN = 5.0


def _elapsed():
    global _START
    if _START is None:
        _START = _time.time()
    return _time.time() - _START


def _fast_fallback(ready_operations):
    # O(n) greedy: most remaining work, then shortest duration.
    best_i = 0
    best_key = None
    for i, op in enumerate(ready_operations):
        mw = op["remaining_work"]
        opts = op["machine_options"]
        min_d = min(opts.values())
        key = (mw, -min_d)
        if best_key is None or key > best_key:
            best_key = key
            best_i = i
    return best_i


def _full_heuristic(ready_operations, time):
    """
    Composite weighted dispatch:
      - Primary: maximize remaining_work (long jobs first).
      - Secondary: minimize best (shortest) duration across machines.
      - Tertiary: minimize remaining_ops (finish jobs to free successors).
      - Quaternary: minimize contention on chosen machine.
    Returns (index, machine).
    """
    n = len(ready_operations)
    if n == 0:
        return (0, 0)
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: opts[m])
        return (0, best_m)

    # Precompute machine contention: how many ready ops can use each machine.
    machine_load = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_load[m] = machine_load.get(m, 0) + 1

    # Weights for composite score (higher is better)
    # Tuned to balance work priority vs. duration vs. completion vs. contention
    W_WORK = 1.0
    W_DUR = 0.3
    W_OPS = 0.1
    W_CONT = 0.05

    # Normalize remaining_work to [0,1] for better weight balance
    all_work = [op["remaining_work"] for op in ready_operations]
    min_work = min(all_work)
    max_work = max(all_work)
    work_range = max_work - min_work if max_work > min_work else 1.0

    # Normalize durations: collect all min durations
    min_durs = []
    for op in ready_operations:
        min_durs.append(min(op["machine_options"].values()))
    min_dur_min = min(min_durs)
    min_dur_max = max(min_durs)
    dur_range = min_dur_max - min_dur_min if min_dur_max > min_dur_min else 1.0

    best_i = 0
    best_score = float('-inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        mw = op["remaining_work"]
        ro = op["remaining_ops"]
        opts = op["machine_options"]

        # Find the best machine for this op: shortest duration, tie-break by lower contention
        min_d = None
        min_m = None
        for m, d in opts.items():
            if min_d is None or d < min_d or (d == min_d and machine_load.get(m, 0) < machine_load.get(min_m, 0)):
                min_d = d
                min_m = m

        cont = machine_load.get(min_m, 0)

        # Normalized values
        norm_work = (mw - min_work) / work_range
        norm_dur = (min_d - min_dur_min) / dur_range  # lower is better, so negate

        # Composite score: higher is better
        score = (
            W_WORK * norm_work
            - W_DUR * norm_dur
            - W_OPS * ro
            - W_CONT * cont
        )

        if score > best_score:
            best_score = score
            best_i = i
            best_machine = min_m

    # Final machine selection for the chosen op with contention-aware tie-breaking
    best_op = ready_operations[best_i]
    opts = best_op["machine_options"]
    final_m = best_machine
    final_d = opts.get(best_machine, float('inf'))
    for m, d in opts.items():
        if d < final_d - 1e-9:
            final_m = m
            final_d = d
        elif abs(d - final_d) <= 1e-9:
            if machine_load.get(m, 0) < machine_load.get(final_m, 0):
                final_m = m

    return (best_i, final_m)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    # deadline guard: if we're close to the limit, use the fast path
    if _elapsed() > (_DEADLINE - _MARGIN):
        return _fast_fallback(ready_operations)
    # full heuristic
    return _full_heuristic(ready_operations, time)
# EVOLVE-BLOCK-END
