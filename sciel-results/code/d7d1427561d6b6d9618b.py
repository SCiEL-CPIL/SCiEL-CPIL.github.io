"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def _bottom_level(times, preds):
    """Compute bottom-level (longest path to a sink) for each task.

    times: array of task durations, indexed by task id.
    preds: list where preds[i] is the list of immediate predecessors of task i.
    Returns an array bl where bl[i] = longest path length from i to any sink,
    including i's own duration.
    """
    n = len(times)
    # Build successors list.
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)

    # Topological order (Kahn's algorithm).
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            indeg[i] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # If graph has a cycle, fall back to original order.
    if len(topo) != n:
        topo = list(range(n))

    # Compute bl in reverse topological order.
    bl = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in succs[u]:
            if bl[v] > best:
                best = bl[v]
        bl[u] = times[u] + best
    return bl, succs


def priority(instance):
    t0 = _time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    # Compute bottom-level rank and successor counts.
    bl, succs = _bottom_level(times, preds)
    succ_count = np.array([len(s) for s in succs], dtype=float)

    # Combine: bottom-level dominates, then task time, then successor count.
    # Scale so that bl is the primary key.
    scale = max(float(np.max(bl)) if n else 1.0, 1.0)
    score = bl + 0.001 * times + 0.000001 * succ_count

    if _time.time() - t0 > DEADLINE:
        # Fallback to longest-task-first.
        return times

    return score
# EVOLVE-BLOCK-END
