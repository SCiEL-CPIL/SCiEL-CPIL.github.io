"""Open-shop dispatch: Load-Balancing Water-Filling heuristic."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Track elapsed time to guard against the 150s wall-clock limit.
_start_time = _time.time()
_DEADLINE_FRACTION = 0.90  # safety margin: stop searching at 90% of budget
_DEADLINE_SECONDS = 150.0 * _DEADLINE_FRACTION


def _time_left():
    """Return seconds remaining before the safety deadline."""
    return _DEADLINE_SECONDS - (_time.time() - _start_time)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a Load-Balancing Water-Filling
    heuristic combined with SPT machine selection.

    Strategy:
      1. For each ready operation, identify the best machine (SPT: shortest processing time).
      2. Score each candidate by how much it increases the maximum machine load
         (L-infinity norm), with a smooth surrogate (log-sum-exp) for the max.
      3. Prefer operations that minimally skew the global workload distribution,
         while also factoring in remaining work (to avoid starving long jobs).

    Returns:
        int index of the chosen ready operation.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Gather machine loads from the 'time' parameter if available,
    # otherwise estimate from remaining_work.
    # 'time' is the current simulation time; we don't have direct machine load info,
    # so we use a proxy: track estimated machine loads from operation durations.
    # Since we only see ready_operations, we estimate load balance from
    # the machine_options durations and remaining_work.

    # For each operation, find the best machine via SPT (shortest duration).
    # Score = combination of:
    #   - Increase in max machine load (water-filling penalty)
    #   - Remaining work (to prioritize jobs with more total work)
    #   - Duration on best machine (prefer shorter to free machines quickly)

    best_idx = 0
    best_score = float('inf')

    # Estimate current machine loads from the ready operations' machine_options.
    # We don't have explicit machine loads, so we use a heuristic:
    # Assume machines are roughly balanced; the penalty for assigning to a machine
    # is proportional to how much work that machine already has.
    # Since we can't observe machine loads directly, we use the sum of durations
    # across all ready operations per machine as a proxy for load.

    # Build a proxy machine load vector.
    # machine_options is {machine: duration} for each op.
    # We'll sum durations per machine across all ready ops as a load estimate.
    machine_load = {}
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            machine_load[m] = machine_load.get(m, 0.0) + dur

    # Normalize machine loads for scoring.
    max_load = max(machine_load.values()) if machine_load else 1.0
    if max_load == 0:
        max_load = 1.0

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op["machine_options"]

        if not machine_options:
            continue

        # SPT: pick the machine with the shortest duration for this op.
        best_machine = min(machine_options, key=lambda m: machine_options[m])
        spt_duration = machine_options[best_machine]

        # Water-filling penalty: how much does assigning this op to best_machine
        # increase the max machine load? Use log-sum-exp for smooth max.
        # Current max load proxy:
        # After assignment, best_machine's load increases by spt_duration.
        # We want to minimize the increase in the L-infinity norm.
        # Smooth surrogate: use log-sum-exp of (load + assigned_duration) / load.

        # Compute a smooth max penalty:
        # For each machine, compute (load_m + (dur if m == best_machine else 0))
        # and take a soft-max via log-sum-exp.
        loads = []
        for m, load in machine_load.items():
            new_load = load + (spt_duration if m == best_machine else 0.0)
            loads.append(new_load / max_load)  # normalize

        # log-sum-exp for smooth max.
        # Use a sharper temperature (divide by tau) to make the surrogate
        # track the true max more closely while staying smooth.
        # tau < 1 => sharper peak => closer to hard max.
        tau = 0.5
        if loads:
            scaled = [x / tau for x in loads]
            max_val = max(scaled)
            if max_val > 0:
                lse = max_val + math.log(sum(math.exp(x - max_val) for x in scaled))
            else:
                lse = 0.0
        else:
            lse = 0.0

        # Combine scores:
        # Primary: minimize lse (water-filling penalty)
        # Secondary: prefer higher remaining_work (MWR) to avoid starvation
        # Tertiary: prefer shorter SPT duration to free machines
        # Quaternary: scarcity bonus — prefer ops whose best machine is less
        #   contested by other ready ops (fewer alternatives on that machine).
        # Weighting: lse dominates; remaining_work breaks ties; scarcity is a
        #   small nudge to avoid overloading a single machine.
        #
        # Count how many ready ops have the same best_machine as this op.
        # Higher count => more contention => slightly penalize.
        contention = 0
        for other in ready_operations:
            om = other["machine_options"]
            if om:
                om_best = min(om, key=lambda m: om[m])
                if om_best == best_machine:
                    contention += 1

        # Normalize contention to [0,1] over n ops.
        contention_norm = contention / max(n, 1)

        score = (lse
                 - 0.15 * (remaining_work / (max_load + 1e-9))
                 + 0.05 * (spt_duration / max_load)
                 + 0.08 * contention_norm)

        if score < best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
