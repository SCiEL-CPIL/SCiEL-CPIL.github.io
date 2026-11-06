"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)

    # Compute total (transitive) successor count for each task
    total_successors = np.zeros(n, dtype=int)
    for i in range(n):
        visited = set()
        stack = list(successors[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(successors[node])
        total_successors[i] = len(visited)

    # Normalize to [0, 1]
    max_time = times.max() if times.max() > 0 else 1.0
    max_succ = total_successors.max() if total_successors.max() > 0 else 1.0

    norm_time = times / max_time
    norm_succ = total_successors.astype(float) / max_succ

    # Combined priority: emphasize successor count (bottleneck) and task time
    # Higher priority = placed earlier in stations
    priority_scores = 0.3 * norm_time + 0.7 * norm_succ

    # Deterministic tie-breaker
    priority_scores = priority_scores + np.arange(n) * 1e-9

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return times  # Fallback to simple time-based

    return priority_scores
# EVOLVE-BLOCK-END
