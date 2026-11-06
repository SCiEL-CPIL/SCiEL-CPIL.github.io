"""Job-Shop dispatch via Look-Ahead Critical-Path Reduction with Conflict + Load Balancing."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _min_pair(machine_options):
    """Return (min_duration, machine) from {machine: duration}."""
    if not machine_options:
        return 1.0, None
    best_m, best_d = None, float("inf")
    for m, d in machine_options.items():
        if d < best_d:
            best_d = d
            best_m = m
    return best_d, best_m


def _is_time_up():
    return (_time.time() - _START_TIME) > _TIME_LIMIT


def dispatch(ready_operations, time):
    """
    Look-ahead dispatch rule.

    For each ready operation we simulate its immediate completion on its best
    (shortest) machine and measure how much the global makespan lower bound
    (max remaining work across all jobs) drops. We then penalise choices that
    would block other ready operations on the same machine (conflict awareness)
    AND penalise choices that push an already-heavily-loaded machine further
    toward the bottleneck (load balancing). The operation that yields the
    greatest net reduction in the makespan lower bound (after the conflict and
    load-balance penalties) is selected.

    The load-balance term models the job-shop as a bipartite graph between
    ready operations and machines: it explicitly avoids machine bottlenecks
    that purely job-centric heuristics ignore.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0

    n = len(ready_operations)

    if n == 1:
        return 0

    # ---- Precompute per-op best machine/duration and per-job remaining work ----
    op_min_dur = [0.0] * n
    op_min_mach = [None] * n
    job_remaining = {}
    for i, op in enumerate(ready_operations):
        d, m = _min_pair(op["machine_options"])
        op_min_dur[i] = d
        op_min_mach[i] = m
        j = op["job"]
        job_remaining[j] = job_remaining.get(j, 0.0) + op["remaining_work"]

    # ---- Current makespan lower bound = max remaining work across jobs ----
    current_lb = max(job_remaining.values()) if job_remaining else 0.0
    if current_lb <= 0:
        current_lb = 1.0

    # ---- Machine load tracking: total ready work that would land on each machine ----
    machine_load = {}
    for i, op in enumerate(ready_operations):
        m = op_min_mach[i]
        if m is None:
            continue
        d = op_min_dur[i]
        machine_load[m] = machine_load.get(m, 0.0) + d

    # Max machine load (a proxy for the makespan lower bound on the machine side)
    max_machine_load = max(machine_load.values()) if machine_load else 0.0
    if max_machine_load <= 0:
        max_machine_load = 1.0

    best_idx = 0
    best_score = -float("inf")

    for i in range(n):
        d = op_min_dur[i]
        m = op_min_mach[i]
        j = ready_operations[i]["job"]

        # ---- Simulate completion: this job's remaining work drops by d ----
        new_job_remaining = job_remaining[j] - d
        if new_job_remaining < 0:
            new_job_remaining = 0.0

        # New makespan lower bound = max over all jobs of (remaining - d if job==j)
        if new_job_remaining >= current_lb:
            new_lb = new_job_remaining
        else:
            # The max was this job before; recompute among others
            new_lb = 0.0
            for jj, rr in job_remaining.items():
                if jj == j:
                    continue
                if rr > new_lb:
                    new_lb = rr
            if new_job_remaining > new_lb:
                new_lb = new_job_remaining

        # Reduction in the makespan lower bound (higher = better)
        reduction = current_lb - new_lb

        # ---- Conflict penalty: work on the same machine that we would delay ----
        if m is not None:
            total_on_m = machine_load.get(m, 0.0)
            other_block = total_on_m - d  # exclude our own duration
        else:
            other_block = 0.0

        # Scale conflict penalty relative to the lower bound so it is comparable
        # to the reduction term. A blocked long op is more harmful.
        conflict_penalty = 0.5 * (other_block / current_lb)

        # ---- Load-balance penalty: how much does this op push the max machine load? ----
        # If we pick an op on machine m, machine m's load becomes (total_on_m).
        # The increase in the max machine load is how much the new max exceeds the
        # old max (excluding m). This explicitly avoids bottlenecking a machine.
        if m is not None:
            total_on_m = machine_load.get(m, 0.0)
            # Max load on OTHER machines (excluding m)
            other_max = 0.0
            for mm, ll in machine_load.items():
                if mm == m:
                    continue
                if ll > other_max:
                    other_max = ll
            # New max machine load if we assign this op to m
            new_max_machine_load = max(total_on_m, other_max)
            # Increase in max machine load
            load_increase = max(0.0, new_max_machine_load - max_machine_load)
            # Scale relative to the max machine load
            load_penalty = 0.3 * (load_increase / max_machine_load)
        else:
            load_penalty = 0.0

        # ---- Net score: reward reduction, penalise conflict and load imbalance ----
        score = reduction - conflict_penalty - load_penalty

        # Tie-breaker: shorter duration is preferred (less blockage)
        score -= 0.001 * d

        if score > best_score:
            best_score = score
            best_idx = i

        # Deadline guard inside the loop
        if _is_time_up():
            break

    return int(best_idx)
# EVOLVE-BLOCK-END
