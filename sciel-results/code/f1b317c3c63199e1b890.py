"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def _build_graph(n, predecessors):
    """Build adjacency lists for successors."""
    successors = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    return successors, in_degree


def _longest_path_from_task(n, times, successors):
    """
    Compute the longest path duration from each task to any sink.
    Uses topological order (reverse).
    """
    # Compute out-degree for reverse topological sort
    out_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in successors[i]:
            out_degree[i] += 1

    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if out_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        # Reduce out-degree of predecessors... wait, we need predecessors for this.
        # Actually, let's just do a DFS-based memoization or use the fact that
        # in a DAG, we can process nodes in reverse topological order.
        # Let me redo this with a proper topological sort.
        pass

    # Better approach: compute topological order first
    in_degree = np.zeros(n, dtype=int)
    pred_list = defaultdict(list)
    for i in range(n):
        preds = []
        # We need to reconstruct predecessors from the instance
        # Actually, let's pass predecessors explicitly
        pass

    # Let me restructure: compute longest path using DFS with memoization
    memo = np.full(n, -1.0)

    def dfs(node):
        if memo[node] >= 0:
            return memo[node]
        if not successors[node]:
            memo[node] = times[node]
            return memo[node]
        max_succ = 0.0
        for s in successors[node]:
            val = dfs(s)
            if val > max_succ:
                max_succ = val
        memo[node] = times[node] + max_succ
        return memo[node]

    for i in range(n):
        dfs(i)

    return memo


def _compute_priorities(instance):
    """
    Compute priority scores for each task based on downstream slack deficit.
    Priority = cycle_time - (longest path from task to any sink)
    Higher priority means less slack, so should be placed earlier.
    
    We also incorporate task duration and successor count as tiebreakers.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Build successors graph
    successors = defaultdict(list)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)

    # Compute longest path from each task to any sink
    # Using iterative approach to avoid recursion depth issues
    memo = np.full(n, -1.0)

    # Topological sort using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    pred_list = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            pred_list[i].append(p)
            in_degree[p] += 1  # Wait, this is wrong. in_degree should count predecessors.

    # Let me redo this properly
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    # Process in reverse topological order to compute longest path to sink
    longest_to_sink = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if not successors[node]:
            longest_to_sink[node] = times[node]
        else:
            max_succ = max(longest_to_sink[s] for s in successors[node])
            longest_to_sink[node] = times[node] + max_succ

    # Downstream slack deficit = cycle_time - longest_to_sink
    # Higher deficit means more slack, lower deficit means tighter
    # We want to prioritize tasks with LOW slack (high priority)
    # So priority = cycle_time - longest_to_sink (higher is better priority)
    # But we also want to break ties by task duration (longer first) and successor count

    # Primary: downstream slack deficit (higher = more urgent)
    slack_deficit = cycle_time - longest_to_sink

    # Secondary: task duration (longer tasks first as tiebreaker)
    # Tertiary: number of successors (more successors = more branching, place earlier)
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Combine: primary metric dominates, with small weights for tiebreakers
    # Normalize to avoid scale issues
    if n > 0:
        slack_min = np.min(slack_deficit)
        slack_max = np.max(slack_deficit)
        slack_range = slack_max - slack_min if slack_max > slack_min else 1.0
        slack_norm = (slack_deficit - slack_min) / slack_range

        time_min = np.min(times)
        time_max = np.max(times)
        time_range = time_max - time_min if time_max > time_min else 1.0
        time_norm = (times - time_min) / time_range

        succ_min = np.min(succ_count)
        succ_max = np.max(succ_count)
        succ_range = succ_max - succ_min if succ_max > succ_min else 1.0
        succ_norm = (succ_count - succ_min) / succ_range

        # Weighted combination: slack deficit dominates
        priority = 0.7 * slack_norm + 0.25 * time_norm + 0.05 * succ_norm
    else:
        priority = np.zeros(0, dtype=float)

    return priority


def priority(instance):
    """
    Score each assembly task for the greedy station-filling algorithm.
    Returns one score per task; higher score = higher priority.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    try:
        return _compute_priorities(instance)
    except Exception:
        # Fallback: return raw times if anything goes wrong
        try:
            return np.asarray(instance["times"], dtype=float)
        except Exception:
            return []
# EVOLVE-BLOCK-END
