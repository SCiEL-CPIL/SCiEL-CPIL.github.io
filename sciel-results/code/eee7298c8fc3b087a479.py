import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = 140.0  # safety margin before 150s kill

# Persistent machine load tracker
_machine_load = {}


def dispatch(ready_operations, time):
    """Dispatch rule: composite of SPT, remaining work, machine load, and urgency."""
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _time.monotonic() - _START > _DEADLINE - 5.0:
        # Fallback: pick min duration (SPT)
        durs = [min(op.get("machine_options", {0: 1}).values(), default=1) for op in ready_operations]
        return int(np.argmin(durs))

    scores = np.zeros(n, dtype=np.float64)
    min_durs = np.zeros(n, dtype=np.float64)
    rem_work = np.zeros(n, dtype=np.float64)
    rem_ops = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {0: 1})
        durs = list(mo.values())
        d_min = min(durs) if durs else 1.0
        min_durs[i] = d_min

        rw = op.get("remaining_work", 0)
        rem_work[i] = rw if rw > 0 else 1.0

        ro = op.get("remaining_ops", 0)
        rem_ops[i] = ro if ro > 0 else 1.0

        # Find the best machine (min duration) and its current load
        best_machine = min(mo, key=mo.get)
        m_load = _machine_load.get(best_machine, 0.0)

        # Virtual release time: when this op could start given machine load
        virtual_release = time + m_load
        # Deadline: earliest completion if scheduled now
        deadline = virtual_release + d_min

        # Composite score (lower is better):
        # 1. Urgency: operations with more remaining work should be prioritized
        #    Use 1/remaining_work so less remaining work = higher urgency... 
        #    Actually, in job-shop, we want to prioritize jobs with MORE remaining work
        #    (critical path). So score component = 1 / (rem_work + eps)
        # 2. SPT: shorter processing time is better
        # 3. Machine contention: prefer machines with less load
        # 4. Deadline-based: earlier deadline is better (EDF)

        # Normalize-ish components
        urgency = 1.0 / (rem_work[i] + 0.001)
        spt = d_min
        load_penalty = m_load
        deadline_score = deadline

        # Weighted combination
        w_urgency = 0.3
        w_spt = 0.3
        w_load = 0.2
        w_deadline = 0.2

        scores[i] = (w_urgency * urgency +
                     w_spt * spt / (np.max(min_durs) + 0.001) +
                     w_load * load_penalty / (max(_machine_load.values()) + 0.001 if _machine_load else 1.0) +
                     w_deadline * deadline_score / (time + 1.0))

    # Update machine loads for the selected operation
    idx = int(np.argmin(scores))
    op = ready_operations[idx]
    mo = op.get("machine_options", {0: 1})
    best_machine = min(mo, key=mo.get)
    d_min = mo[best_machine]
    _machine_load[best_machine] = _machine_load.get(best_machine, 0.0) + d_min

    return idx
