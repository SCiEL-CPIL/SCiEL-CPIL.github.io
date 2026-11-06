"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import heapq
import time


def _build_successor_adjacency(n, predecessors):
    """Build successor adjacency list from predecessors."""
    succ = [[] for _ in range(n)]
    for j, preds in enumerate(predecessors):
        for i in preds:
            succ[i].append(j)
    return succ


def _topological_order(n, predecessors):
    """Return tasks in topological order (predecessors before successors)."""
    in_degree = [0] * n
    for preds in predecessors:
        for _ in preds:
            pass
    # Build in-degree from predecessors
    for j, preds in enumerate(predecessors):
        in_degree[j] = len(preds)
    
    # Use Kahn's algorithm
    queue = [i for i in range(n) if in_degree[i] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        # Find successors of node
        for j in range(n):
            if node in predecessors[j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    return order


def _compute_longest_path_downstream(n, succ, times):
    """Compute longest path from each task to any sink (including its own time)."""
    # Process in reverse topological order
    top_order = _topological_order(n, [[i for i in range(n) if _is_pred(i, j, n, predecessors)] for j in range(n)])
    # Simpler: just use memoized DFS
    memo = [-1] * n
    
    def dfs(node):
        if memo[node] != -1:
            return memo[node]
        max_down = 0
        for s in succ[node]:
            max_down = max(max_down, dfs(s))
        memo[node] = times[node] + max_down
        return memo[node]
    
    for i in range(n):
        dfs(i)
    return memo


def _is_pred(i, j, n, predecessors):
    return i in predecessors[j]


def _compute_successor_count(n, succ):
    """Count direct successors for each task."""
    return [len(succ[i]) for i in range(n)]


def _compute_comprehensive_priority(instance):
    """Compute priority using multiple heuristics."""
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor adjacency
    succ = _build_successor_adjacency(n, predecessors)
    
    # Compute longest path downstream (including own time)
    memo = [-1] * n
    # Use iterative approach to avoid recursion depth issues
    # Process nodes in reverse topological order
    in_degree = [len(preds) for preds in predecessors]
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        node = queue.pop(0)
        top_order.append(node)
        for j in range(n):
            if node in predecessors[j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    
    # Reverse topological order for downstream computation
    for node in reversed(top_order):
        max_down = 0
        for s in succ[node]:
            max_down = max(max_down, memo[s])
        memo[node] = times[node] + max_down
    
    longest_downstream = np.array(memo, dtype=float)
    
    # Successor count
    succ_count = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Combine heuristics: prioritize tasks with large downstream paths and many successors
    # Normalize each component
    if np.max(longest_downstream) > 0:
        ld_norm = longest_downstream / np.max(longest_downstream)
    else:
        ld_norm = np.zeros(n)
    
    if np.max(succ_count) > 0:
        sc_norm = succ_count / np.max(succ_count)
    else:
        sc_norm = np.zeros(n)
    
    # Weighted combination
    priority = 0.5 * times + 0.3 * ld_norm * np.max(times) + 0.2 * sc_norm * np.max(times)
    
    return priority


def priority(instance):
    """Score each assembly task for priority-based station assignment."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    try:
        return _compute_comprehensive_priority(instance)
    except Exception:
        # Fallback to simple time-based priority
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
