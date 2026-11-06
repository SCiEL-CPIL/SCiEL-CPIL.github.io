"""Open-shop dispatch rule: CPM-proxy with smooth machine load balancing."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def _soft_max(vals, temperature=1.0):
    """Smooth max via log-sum-exp for differentiable surrogate of max."""
    arr = np.array(vals, dtype=float)
    if len(arr) == 0:
        return 0.0
    m = arr.max()
    if m < 1e-12:
        return 0.0
    return m + temperature * np.log(np.sum(np.exp((arr - m) / temperature)))


def _best_machine_for_op(op):
    """Return (machine, duration) for the fastest option of an operation."""
    opts = op["machine_options"]
    best_m = min(opts, key=lambda m: (opts[m], m))
    return best_m, opts[best_m]


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, _ = _best_machine_for_op(op)
        return (0, best_m)

    # Pre-compute per-operation features
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    all_durs = []  # all durations for each op (for smooth load)
    job_ids = []

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m, best_d = _best_machine_for_op(op)
        best_machines.append(best_m)
        min_durs.append(best_d)
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        all_durs.append(list(opts.values()))
        job_ids.append(op["job"])

    # Compute smooth machine load per machine:
    # For each machine, estimate the "load" as soft-max of durations of ops
    # that could use it, weighted by how much they prefer it.
    # This gives a smooth surrogate for the max-duration bottleneck.
    machine_loads = {}
    all_machines = set()
    for m in best_machines:
        all_machines.add(m)
    # Also include machines that appear in any op's options
    for i in range(n):
        for m in ready_operations[i]["machine_options"]:
            all_machines.add(m)

    for m in all_machines:
        # Collect durations of all ops that can use machine m
        durs_on_m = []
        for i in range(n):
            opts = ready_operations[i]["machine_options"]
            if m in opts:
                durs_on_m.append(opts[m])
        if durs_on_m:
            # Soft max: smooth surrogate for the worst-case duration on this machine
            machine_loads[m] = _soft_max(durs_on_m, temperature=0.5)
        else:
            machine_loads[m] = 0.0

    # For each op, compute the "contention cost" of using its best machine:
    # How loaded is that machine relative to alternatives?
    contention_costs = []
    for i in range(n):
        bm = best_machines[i]
        bm_load = machine_loads.get(bm, 0.0)
        # Also compute soft-max load if we were to use other machines
        # The cost is how much this machine's load dominates
        contention_costs.append(bm_load)

    # Deadline guard: fall back to simple deterministic choice
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # CPM proxy: prioritize operations on jobs with the longest remaining work
    # Use a combined tail: remaining_work + alpha * remaining_ops (ops count as
    # sequential stages that each add some processing time)
    alpha = 0.3  # weight for remaining ops in tail estimate
    job_tails = {}
    for i in range(n):
        j = job_ids[i]
        tail = rem_works[i] + alpha * rem_ops[i]
        if j not in job_tails or tail > job_tails[j]:
            job_tails[j] = tail

    job_tail_vals = [job_tails[job_ids[i]] for i in range(n)]

    # Normalize criteria
    def _normalize(vals):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_tail = _normalize(job_tail_vals)            # higher = longer critical path
    norm_contention = _normalize(contention_costs)    # higher = more loaded machine
    norm_min_dur = _normalize(min_durs)               # higher = slower min duration
    norm_rem_ops = _normalize(rem_ops)                # higher = more ops left

    # Weights for CPM-proxy strategy:
    # 1. Job tail (CPM): dominant factor, prioritize longest remaining jobs
    # 2. Machine contention (smooth): avoid bottlenecking heavily-loaded machines
    # 3. Min duration: among tied tails, prefer faster ops to free up resources
    # 4. Remaining ops: prefer ops with fewer remaining stages (shorter tail)
    
    w_tail = 0.45
    w_contention = 0.30
    w_min_dur = 0.15
    w_rem_ops = 0.10

    # Composite score: higher is better to schedule NOW
    scores = (
        w_tail * norm_tail
        + w_contention * (1.0 - norm_contention)
        + w_min_dur * (1.0 - norm_min_dur)
        + w_rem_ops * (1.0 - norm_rem_ops)
    )

    idx = int(np.argmax(scores))

    # Tie-break: if scores are very close, prefer lower min duration, then fewer rem_ops
    top_score = scores[idx]
    close_candidates = [i for i in range(n) if abs(scores[i] - top_score) < 1e-9]
    if len(close_candidates) > 1:
        idx = min(close_candidates, key=lambda i: (min_durs[i], rem_ops[i]))

    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
