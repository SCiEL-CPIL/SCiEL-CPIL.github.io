"""New program for Resource-Constrained Project Scheduling (serial SGS).

Approach: Minimum Total Float (MSTF) Priority.
This uses the Critical Path Method (CPM) to calculate the total float
(slack) for each activity. Activities with the least float (closest to
the critical path) are prioritized. This is a distinct approach from
Bottom-Level or Simple Duration heuristics, as it accounts for the
backward pass and the overall project duration constraints.
"""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Predecessor list per activity.
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    # Normalise predecessors to lists of ints.
    pred_lists = []
    for i in range(n):
        p = preds[i]
        if p is None:
            p = []
        if isinstance(p, (list, tuple, np.ndarray)):
            p = [int(x) for x in p]
        else:
            p = [int(p)]
        # Filter out self and out-of-range.
        p = [x for x in p if 0 <= x < n and x != i]
        pred_lists.append(p)

    # Successor list per activity.
    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in pred_lists[i]:
            succ_lists[j].append(i)

    # ---- Topological Sort (Kahn's Algorithm) ----
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in pred_lists[i]:
            in_degree[i] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Handle potential cycles (should not happen in RCPSP)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(list(remaining))

    # ---- CPM Forward Pass: Earliest Start (ES) and Earliest Finish (EF) ----
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    
    for i in topo_order:
        if pred_lists[i]:
            ES[i] = max(EF[j] for j in pred_lists[i])
        else:
            ES[i] = 0.0
        EF[i] = ES[i] + durations[i]
        
    # Project makespan (unconstrained)
    T = float(np.max(EF)) if n > 0 else 0.0

    # ---- CPM Backward Pass: Latest Finish (LF) and Latest Start (LS) ----
    LF = np.full(n, T, dtype=float)
    LS = np.full(n, T, dtype=float)
    
    # Initialize LF for sinks to T, then propagate backwards
    # We iterate in reverse topological order
    for i in reversed(topo_order):
        # If i has no successors, LF[i] is already T (initialized)
        # If i has successors, LF[i] = min(LS[succ])
        if succ_lists[i]:
            LF[i] = min(LS[j] for j in succ_lists[i])
        # else LF[i] remains T
        
        LS[i] = LF[i] - durations[i]

    # ---- Total Float (TF) ----
    # TF[i] = LS[i] - ES[i]
    TF = LS - ES
    
    # ---- Priority Score ----
    # We want to prioritize activities with LOW float.
    # So higher score = lower float.
    
    # Normalize TF to [0, 1] where 0 is critical (high priority) and 1 is max slack (low priority)
    # Score = 1 - (TF / T)
    if T > 1e-9:
        norm_tf = TF / T
        score = 1.0 - norm_tf
    else:
        score = np.ones(n, dtype=float)
        
    # Add a very small tie-breaker based on duration to prefer shorter tasks in ties?
    # In ties of float, it's often good to schedule shorter tasks to reduce potential delays?
    # Let's add a tiny epsilon based on -duration to prefer shorter tasks.
    dur_max = float(np.max(durations)) if n > 0 else 1.0
    if dur_max > 1e-9:
        dur_norm = durations / dur_max
        score += 0.001 * (1.0 - dur_norm) # Prefer shorter (higher score)
        
    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
