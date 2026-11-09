# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).

    Features:
    1. Critical path length (longest path from activity to end)
    2. Longest path from start
    3. Duration
    4. Resource demand intensity
    5. Successor count
    6. Resource bottleneck: min capacity required for this activity to be on the
       critical resource profile (higher bottleneck => more constraining => prioritize)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Get predecessors/successors
    preds = instance.get("predecessors", None)

    # Build successor list
    succs = [[] for _ in range(n)]
    if preds is not None:
        for i in range(n):
            if i < len(preds):
                for p in preds[i]:
                    if p is not None and 0 <= p < n:
                        succs[p].append(i)

    # Compute longest path from each activity to end (critical path length)
    # Using topological order via Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        if i < len(preds):
            for p in preds[i]:
                if p is not None and 0 <= p < n:
                    in_degree[i] += 1

    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree.copy()

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_degree_copy[s] -= 1
            if in_degree_copy[s] == 0:
                queue.append(s)

    # If not all nodes are in topo_order, add remaining (handle cycles gracefully)
    if len(topo_order) < n:
        seen = set(topo_order)
        for i in range(n):
            if i not in seen:
                topo_order.append(i)

    # Compute longest path from each node to end (reverse topological order)
    longest_to_end = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if succs[node]:
            longest_to_end[node] = durations[node] + np.max([longest_to_end[s] for s in succs[node]])
        else:
            longest_to_end[node] = durations[node]

    # Compute longest path from start to each node
    longest_from_start = np.zeros(n, dtype=float)
    for node in topo_order:
        if preds is not None and node < len(preds) and preds[node]:
            valid_preds = [p for p in preds[node] if p is not None and 0 <= p < n]
            if valid_preds:
                longest_from_start[node] = np.max([longest_from_start[p] + durations[p] for p in valid_preds])

    # Resource demands
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)

        # Resource demand intensity: sum of (demand/capacity) for each resource
        with np.errstate(divide='ignore', invalid='ignore'):
            intensity = np.sum(np.where(capacities > 0, demands / capacities, 0), axis=1)
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)

        # Resource bottleneck: for each activity, compute the minimum capacity
        # required across resources such that the activity's demand would be a
        # significant fraction of capacity. Activities that would be near-capacity
        # under tight scheduling are more constraining.
        # Use: for each activity, max over resources of (demand_r / capacity_r)
        # This identifies which activities most stress the tightest resource.
        with np.errstate(divide='ignore', invalid='ignore'):
            bottleneck = np.max(np.where(capacities > 0, demands / capacities, 0), axis=1)
        bottleneck = np.nan_to_num(bottleneck, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        intensity = np.zeros(n, dtype=float)
        bottleneck = np.zeros(n, dtype=float)

    # Successor count
    succ_count = np.array([len(s) for s in succs], dtype=float)

    # Predecessor count (activities with many predecessors may be delayed)
    pred_count = np.zeros(n, dtype=float)
    if preds is not None:
        for i in range(n):
            if i < len(preds):
                valid = [p for p in preds[i] if p is not None and 0 <= p < n]
                pred_count[i] = len(valid)

    # Normalize each component
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_lte = normalize(longest_to_end)
    norm_lfs = normalize(longest_from_start)
    norm_dur = normalize(durations)
    norm_int = normalize(intensity)
    norm_sc = normalize(succ_count)
    norm_bn = normalize(bottleneck)

    # Weighted combination
    # Critical path length is most important; bottleneck captures resource constraint
    w_lte = 0.30
    w_lfs = 0.10
    w_dur = 0.15
    w_int = 0.10
    w_sc = 0.10
    w_bn = 0.25

    scores = (w_lte * norm_lte +
              w_lfs * norm_lfs +
              w_dur * norm_dur +
              w_int * norm_int +
              w_sc * norm_sc +
              w_bn * norm_bn)

    return scores
# EVOLVE-BLOCK-END
