"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        # pick machine with smallest duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Precompute per-op info
    # For each operation, consider all machine options.
    # In an open shop, each operation must be processed on exactly one machine
    # (or a subset). The "machine_options" dict maps machine -> duration.
    # We pick which operation to schedule next, and optionally which machine.

    # Heuristic scoring:
    # 1. Prioritize operations with more remaining work (longest remaining processing time)
    #    to reduce makespan via a Longest Remaining Processing Time (LRPT) style rule.
    # 2. For machine selection, choose the machine that minimizes the resulting
    #    completion time considering that machine's current load (if known) or
    #    simply the shortest duration to keep the operation short.
    # 3. Break ties by fewer remaining operations (finish jobs sooner to free up
    #    resources earlier).
    # 4. Additional tie-break: lower job id for determinism.

    # Compute per-op info: min duration, best machine, and flexibility (num options)
    min_durs = []
    best_machines = []
    flexibilities = []
    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        flexibilities.append(len(opts))

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = max(range(n), key=lambda i: ready_operations[i]["remaining_work"])
        return (idx, best_machines[idx])

    # Composite scoring: prioritize operations that are both urgent (high remaining work)
    # and inflexible (few machine options), while favoring shorter processing times.
    # Score = -remaining_work * (1 + 0.3 * (1/flexibility)) + 0.5 * min_duration
    # Lower score = better candidate.
    # Rationale: 
    #   - High remaining_work means this job has a lot of work left; scheduling it now
    #     prevents it from becoming a long tail.
    #   - Low flexibility (few machine options) means this op is more constrained; 
    #     delaying it risks blocking the job.
    #   - Short min_duration means we can complete this op quickly, freeing resources.
    # The product term weights urgency by criticality, while the additive duration
    # term provides a secondary preference for short ops.
    scores = []
    for i in range(n):
        rw = ready_operations[i]["remaining_work"]
        fl = max(1, flexibilities[i])
        md = min_durs[i]
        # Urgency weighted by inverse flexibility (more constrained = more urgent)
        urgency = rw * (1.0 + 0.3 * (1.0 / fl))
        # Shorter ops are preferred (additive penalty for long ops)
        duration_penalty = 0.5 * md
        # Fewer remaining ops = closer to completion, slight bonus
        rem_ops_bonus = -0.1 * ready_operations[i]["remaining_ops"]
        scores.append(urgency + duration_penalty + rem_ops_bonus)
    
    # Pick lowest score (most urgent/critical)
    idx = min(range(n), key=lambda i: (scores[i], ready_operations[i]["job"]))
    
    # Machine selection: among the chosen op's options, pick the one that minimizes
    # completion time. Since we don't have machine load info, use shortest duration.
    # But also consider: if there's a tie in duration, prefer lower machine id.
    op = ready_operations[idx]
    opts = op["machine_options"]
    best_m = min(opts, key=lambda m: (opts[m], m))
    return (idx, best_m)
# EVOLVE-BLOCK-END
