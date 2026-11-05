"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    works = np.empty(n, dtype=float)
    ops = np.empty(n, dtype=float)
    min_procs = np.empty(n, dtype=float)
    for i, op in enumerate(ready_operations):
        works[i] = float(op["remaining_work"])
        ops[i] = float(op["remaining_ops"])
        mo = op["machine_options"]
        if mo:
            min_procs[i] = float(min(mo.values()))
        else:
            min_procs[i] = 0.0

    ops_safe = np.maximum(ops, 1.0)
    avgs = works / ops_safe

    max_work = float(np.max(works))
    if max_work <= 0:
        max_work = 1.0
    max_avg = float(np.max(avgs))
    if max_avg <= 0:
        max_avg = 1.0
    max_proc = float(np.max(min_procs))
    if max_proc <= 0:
        max_proc = 1.0

    work_norm = works / max_work
    avg_norm = avgs / max_avg
    urgency = 1.0 / ops_safe
    max_urgency = float(np.max(urgency))
    if max_urgency <= 0:
        max_urgency = 1.0
    urgency_norm = urgency / max_urgency
    proc_norm = min_procs / max_proc

    w_work = 0.55
    w_avg = 0.25
    w_urgency = 0.10
    w_proc = 0.10

    scores = (w_work * work_norm +
              w_avg * avg_norm +
              w_urgency * urgency_norm +
              w_proc * proc_norm)

    # Deterministic tie-break: np.argmax returns the lowest index on ties.
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
