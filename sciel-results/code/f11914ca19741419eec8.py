"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as time_module

# Deadline guard variables
_START_TIME = None
_DEADLINE = 110.0  # Safety margin before 120s limit

# Dynamic machine-load model:
# - machine_load[m] = total work scheduled/completed on m so far (approximates when m becomes free if no current work)
# - running_on_m[m] = work currently running on m (started but not yet completed)
# We approximate machine availability as:
#   available_at(m) = max(current_time, machine_load[m] - running_on_m[m]) + running_on_m[m]
# which simplifies to machine_load[m] when current_time <= machine_load[m]
# and to current_time + running_on_m[m] when current_time > machine_load[m] - running_on_m[m].
# This is a lightweight proxy for a true event-driven machine schedule without needing full event logs.
_machine_load = {}
_running_on_m = {}

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time_module.time()

def dispatch(ready_operations, time):
    """
    Dispatches the next ready operation using a bottleneck-aware heuristic:
    - For each ready op, estimate earliest completion on each eligible machine using dynamic load.
    - Route the op to the machine with the earliest completion.
    - Score ops by (earliest_completion, remaining_ops) to prefer ops that finish soon and
      have more downstream work (higher urgency/critical-path weight).
    """
    _init_deadline()

    if not ready_operations:
        return 0

    # If we are close to the deadline, return a simple fallback to avoid complex calculation overhead
    if time_module.time() - _START_TIME > _DEADLINE:
        return 0

    best_score = float('inf')
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0)

        # Estimate earliest completion on each eligible machine
        best_earliest = float('inf')
        best_m_for_op = None

        for m, d in machine_options.items():
            if d is None:
                continue
            d = float(d)
            if d <= 0:
                d = 1e-9

            # Get or initialize load state for this machine
            load = _machine_load.get(m, 0.0)
            running = _running_on_m.get(m, 0.0)

            # Approximate availability:
            # If machine is currently busy with 'running' work, it will be free at (load - running) + running = load.
            # If current time is beyond that free time, machine is idle and available now, but still has running work? 
            # In a real sim, running work would have completed; our model is approximate.
            # We use max(time, load - running) as the time the machine becomes free of prior work,
            # then add running if we assume current work still occupies it until load.
            free_at = max(time, load - running)
            earliest = free_at + d

            if earliest < best_earliest:
                best_earliest = earliest
                best_m_for_op = m

        if best_m_for_op is None:
            continue

        # Score: minimize earliest completion; break ties by preferring more remaining ops (critical path)
        # Add a tiny urgency term to avoid pure tie-breaking on earliest only.
        score = best_earliest - 1e-6 * remaining_ops

        if score < best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Update dynamic load model for the chosen op/machine
    if best_machine is not None:
        m = best_machine
        d = float(ready_operations[best_index].get("machine_options", {}).get(m, 1.0))
        if d <= 0:
            d = 1e-9

        # If machine is idle at current time, it will be busy until time + d
        # If machine is busy, we assume the new op queues after current work, so load increases by d
        # running_on_m is updated to reflect current work if machine was idle, otherwise keep existing running
        load = _machine_load.get(m, 0.0)
        running = _running_on_m.get(m, 0.0)

        # Determine if machine is currently busy: if load > time, there is work extending beyond current time
        if load > time:
            # Machine is busy; new op queues after current work
            _machine_load[m] = load + d
            # running remains the same (current work still running)
        else:
            # Machine is idle; new op starts now and runs for d
            _machine_load[m] = time + d
            _running_on_m[m] = d

    return (best_index, best_machine)
# EVOLVE-BLOCK-END
