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

    # Compute transitive successor sets using BFS/DFS from each node
    # Also compute:
    #   rank[i] = longest path length (in edges) from i to any sink
    #   succ_time_sum[i] = sum of times of all transitive successors of i
    #   succ_count[i] = number of transitive successors

    succ_count = np.zeros(n, dtype=int)
    succ_time_sum = np.zeros(n, dtype=float)
    rank = np.zeros(n, dtype=int)  # longest path from i to sink

    # Process nodes in reverse topological order for rank computation
    # First, get topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1  # p is a predecessor of i, so i has an incoming edge

    # Actually let's be careful: preds[i] are predecessors of task i.
    # Edge direction: predecessor -> successor
    # in_degree[i] = number of predecessors of i
    # Let's redo:
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(preds[i])

    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # If topo_order doesn't have all nodes, there's a cycle; fallback
    if len(topo_order) != n:
        # Fallback: just use time-based priority
        return times.copy()

    # Process in reverse topological order to compute rank and aggregate successor info
    # rank[i] = max(0, max over successors s of (1 + rank[s]))
    # succ_count[i] = 1 + sum over direct successors... no, transitive
    # Better: use the fact that in reverse topo order, all successors are already processed

    # For transitive successor count and time sum, we need to be more careful
    # Let's use a different approach: for each node, compute its transitive successors
    # using the DAG structure

    # Alternative: compute "position" based on longest path from start
    # and "depth" based on longest path to end

    # Longest path from source (start depth)
    start_depth = np.zeros(n, dtype=float)
    for node in topo_order:
        for succ in successors[node]:
            start_depth[succ] = max(start_depth[succ], start_depth[node] + times[node])

    # Longest path to sink (end depth) - process in reverse topo
    end_depth = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        for succ in successors[node]:
            end_depth[node] = max(end_depth[node], times[node] + end_depth[succ])

    # Transitive successor count via DP in reverse topo
    # succ_count[i] = number of transitive successors
    # We can compute this by: for each node, its transitive successors = union of
    # direct successors and their transitive successors
    # But this is O(n^2) worst case. For n=30, it's fine.

    # Use bitsets for efficiency
    succ_bitset = [0] * n  # bitmask of transitive successors
    for node in reversed(topo_order):
        mask = 0
        for succ in successors[node]:
            mask |= (1 << succ) | succ_bitset[succ]
        succ_bitset[node] = mask
        succ_count[node] = bin(mask).count('1')

    # Sum of times of transitive successors
    for node in range(n):
        mask = succ_bitset[node]
        total = 0.0
        for j in range(n):
            if mask & (1 << j):
                total += times[j]
        succ_time_sum[node] = total

    # Check deadline
    if time.time() - t0 > DEADLINE:
        return times.copy()

    # Now compute priority scores
    # Features:
    # 1. end_depth: how much work remains after this task (longest path to sink)
    # 2. succ_count: number of transitive successors (breadth of impact)
    # 3. succ_time_sum: total time of all transitive successors
    # 4. times[i]: the task's own duration

    # Normalize each feature to [0, 1]
    def normalize(x):
        mx = x.max()
        if mx > 0:
            return x / mx
        return x

    norm_end_depth = normalize(end_depth)
    norm_succ_count = normalize(succ_count.astype(float))
    norm_succ_time = normalize(succ_time_sum)
    norm_time = normalize(times)

    # Combined priority:
    # - end_depth captures the "critical path" remaining work
    # - succ_count captures how many tasks depend on this one
    # - succ_time_sum captures the total workload unlocked
    # - time captures the task's own duration

    # Weighting: emphasize end_depth and succ_count as primary drivers
    # The greedy algorithm places highest-priority eligible tasks first.
    # We want to place tasks that unlock the most critical work first.

    priority_scores = (
        0.40 * norm_end_depth +
        0.25 * norm_succ_count +
        0.20 * norm_succ_time +
        0.15 * norm_time
    )

    # Add a small interaction term: tasks that are both on the critical path
    # AND unlock many successors get a bonus
    interaction = norm_end_depth * norm_succ_count
    priority_scores += 0.05 * interaction

    # Deterministic tie-breaker (very small)
    priority_scores = priority_scores + np.arange(n) * 1e-9

    # Final deadline guard
    if time.time() - t0 > DEADLINE:
        return times.copy()

    return priority_scores
# EVOLVE-BLOCK-END
