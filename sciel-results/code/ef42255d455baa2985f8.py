"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and optionally which machine runs it.
    
    Strategy: Bottleneck mitigation via constraint tightness with work-load awareness.
    1. Prioritize operations with the fewest remaining machine options
       (highest constraint tightness) to prevent future starvation.
    2. Break ties by preferring operations with more remaining work (critical path).
    3. For machine selection, balance shortest duration against load distribution.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if len(machine_options) == 1:
            return (0, list(machine_options.keys())[0])
        # Pick machine with shortest duration
        best_m = min(machine_options, key=lambda m: machine_options[m])
        return (0, best_m)

    # Compute constraint tightness: fewer machine options = tighter constraint
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        num_machines = len(machine_options)
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0)

        # Primary: constraint tightness (fewer options = higher priority)
        # Use inverse so fewer options -> higher score
        tightness_score = 1.0 / max(num_machines, 1)

        # Secondary: remaining work (more work = higher priority)
        work_score = remaining_work

        # Tertiary: remaining ops count (more ops left = more critical)
        ops_score = remaining_ops

        # Combine: tightness dominates, then work, then ops
        scores.append((tightness_score, work_score, ops_score))

    # Find the best operation by lexicographic order (tightness, work, ops)
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        s = scores[i]
        # Lexicographic comparison: higher is better
        if s[0] > best_score[0] or \
           (s[0] == best_score[0] and s[1] > best_score[1]) or \
           (s[0] == best_score[0] and s[1] == best_score[1] and s[2] > best_score[2]):
            best_idx = i
            best_score = s

    # Select the best machine for the chosen operation
    op = ready_operations[best_idx]
    machine_options = op["machine_options"]

    if len(machine_options) == 1:
        return (best_idx, list(machine_options.keys())[0])

    # For machine selection: prefer the machine with the shortest processing time
    # This minimizes the critical path contribution of this operation
    best_machine = min(machine_options, key=lambda m: machine_options[m])

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
