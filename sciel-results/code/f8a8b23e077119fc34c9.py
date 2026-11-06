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

    # Compute in-degrees for topological ordering
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(preds[i])

    # Compute rank (longest path to any sink) using reverse topological order
    # rank[i] = max over successors s of (1 + rank[s]), or 0 if no successors
    rank = np.zeros(n, dtype=float)
    # Process nodes in reverse topological order (sinks first)
    # Use Kahn's algorithm variant
    from collections import deque
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    # We need reverse topo order: process sinks first
    # Let's compute out-degree
    out_degree = np.zeros(n, dtype=int)
    for i in range(n):
        out_degree[i] = len(successors[i])
    
    # Reverse Kahn: start from sinks (out_degree == 0)
    rev_q = deque()
    rev_in_degree = in_degree.copy()  # We'll decrement as we process
    # Actually, let's just do a DFS/BFS from each node to compute rank
    # Since n=30, this is cheap
    
    def compute_rank(node, memo={}):
        if node in memo:
            return memo[node]
        if not successors[node]:
            memo[node] = 0.0
            return 0.0
        max_succ_rank = 0.0
        for s in successors[node]:
            sr = compute_rank(s, memo)
            max_succ_rank = max(max_succ_rank, sr)
        memo[node] = 1.0 + max_succ_rank
        return memo[node]
    
    rank_memo = {}
    for i in range(n):
        compute_rank(i, rank_memo)
    rank = np.array([rank_memo.get(i, 0.0) for i in range(n)])
    
    # Compute total transitive successor count
    total_successors = np.zeros(n, dtype=int)
    def count_successors(node, visited=None):
        if visited is None:
            visited = set()
        count = 0
        for s in successors[node]:
            if s not in visited:
                visited.add(s)
                count += 1 + count_successors(s, visited)
        return count
    
    for i in range(n):
        total_successors[i] = count_successors(i)
    
    # Compute weighted successor time sum (sum of times of all transitive successors)
    succ_time_sum = np.zeros(n, dtype=float)
    def sum_succ_times(node, visited=None):
        if visited is None:
            visited = set()
        total = 0.0
        for s in successors[node]:
            if s not in visited:
                visited.add(s)
                total += times[s] + sum_succ_times(s, visited)
        return total
    
    for i in range(n):
        succ_time_sum[i] = sum_succ_times(i)

    # Normalize features
    max_time = times.max() if times.max() > 0 else 1.0
    max_rank = rank.max() if rank.max() > 0 else 1.0
    max_succ = total_successors.max() if total_successors.max() > 0 else 1.0
    max_succ_time = succ_time_sum.max() if succ_time_sum.max() > 0 else 1.0

    norm_time = times / max_time
    norm_rank = rank / max_rank
    norm_succ = total_successors.astype(float) / max_succ
    norm_succ_time = succ_time_sum / max_succ_time

    # Compute critical path length: longest path from source to this task (upstream depth)
    # This captures tasks that are on the critical path and should be scheduled early
    upstream_depth = np.zeros(n, dtype=float)
    # Process in topological order
    topo_order = []
    from collections import deque
    q = deque()
    in_deg = in_degree.copy()
    for i in range(n):
        if in_deg[i] == 0:
            q.append(i)
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                q.append(s)
    
    for node in topo_order:
        if preds[node]:
            upstream_depth[node] = 1.0 + max(upstream_depth[p] for p in preds[node])
        else:
            upstream_depth[node] = 0.0
    
    max_upstream = upstream_depth.max() if upstream_depth.max() > 0 else 1.0
    norm_upstream = upstream_depth / max_upstream
    
    # Compute "bottleneck score": ratio of successor time sum to cycle time
    # Tasks that block a large fraction of remaining work are more critical
    cycle_time = instance.get("cycle_time", 40.0)
    bottleneck_ratio = succ_time_sum / (cycle_time + 1e-9)
    max_bottleneck = bottleneck_ratio.max() if bottleneck_ratio.max() > 0 else 1.0
    norm_bottleneck = bottleneck_ratio / max_bottleneck
    
    # Combined priority with refined weights:
    # - successor time sum (bottleneck): most important - tasks blocking most work
    # - rank (downstream depth): tasks with long chains downstream
    # - upstream depth: tasks on critical path from sources
    # - successor count: number of tasks that depend on this one
    # - time: longer tasks (secondary)
    priority_scores = (
        0.35 * norm_succ_time +
        0.25 * norm_rank +
        0.20 * norm_upstream +
        0.15 * norm_bottleneck +
        0.05 * norm_time
    )

    # Add small deterministic tie-breaker to ensure stable ordering
    priority_scores = priority_scores + np.arange(n) * 1e-10

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return times  # Fallback to simple time-based

    return priority_scores
# EVOLVE-BLOCK-END
