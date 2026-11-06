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
    rank = np.zeros(n, dtype=float)
    
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
    
    # Compute total transitive predecessor count
    total_preds = np.zeros(n, dtype=int)
    def count_preds(node, visited=None):
        if visited is None:
            visited = set()
        count = 0
        for p in preds[node]:
            if p not in visited:
                visited.add(p)
                count += 1 + count_preds(p, visited)
        return count
    
    for i in range(n):
        total_preds[i] = count_preds(i)
    
    # Compute weighted predecessor time sum (sum of times of all transitive predecessors)
    pred_time_sum = np.zeros(n, dtype=float)
    def sum_pred_times(node, visited=None):
        if visited is None:
            visited = set()
        total = 0.0
        for p in preds[node]:
            if p not in visited:
                visited.add(p)
                total += times[p] + sum_pred_times(p, visited)
        return total
    
    for i in range(n):
        pred_time_sum[i] = sum_pred_times(i)
    
    # Resource contention metric: sum of cycle-time-normalized durations of all transitive
    # predecessors and successors, prioritizing tasks in time-dense interconnected subgraphs
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Normalize by cycle time to make metric scale-invariant
    norm_succ_time_ct = succ_time_sum / cycle_time
    norm_pred_time_ct = pred_time_sum / cycle_time
    
    # Resource contention: total temporal weight of the dependency neighborhood
    resource_contention = norm_succ_time_ct + norm_pred_time_ct
    
    # Also compute interaction: tasks that are both upstream-heavy AND downstream-heavy
    # are in critical paths and should be prioritized
    contention_interaction = np.sqrt(norm_succ_time_ct * norm_pred_time_ct)
    
    # Normalize features
    max_time = times.max() if times.max() > 0 else 1.0
    max_rank = rank.max() if rank.max() > 0 else 1.0
    max_succ = total_successors.max() if total_successors.max() > 0 else 1.0
    max_succ_time = succ_time_sum.max() if succ_time_sum.max() > 0 else 1.0
    max_contention = resource_contention.max() if resource_contention.max() > 0 else 1.0
    
    norm_time = times / max_time
    norm_rank = rank / max_rank
    norm_succ = total_successors.astype(float) / max_succ
    norm_succ_time = succ_time_sum / max_succ_time
    norm_contention = resource_contention / max_contention
    norm_interaction = contention_interaction / (max_contention * 0.5) if max_contention > 0 else 0
    
    # Combined priority emphasizing resource contention:
    # - resource_contention: total temporal weight of dependency neighborhood (most important)
    # - contention_interaction: geometric mean of upstream/downstream temporal weight
    # - succ_time: downstream work this task gates
    # - rank: depth in precedence chain
    # - time: task's own duration
    priority_scores = (
        0.40 * norm_contention +
        0.20 * norm_interaction +
        0.15 * norm_succ_time +
        0.15 * norm_rank +
        0.10 * norm_time
    )

    # Add small deterministic tie-breaker to ensure stable ordering
    priority_scores = priority_scores + np.arange(n) * 1e-10

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return times  # Fallback to simple time-based

    return priority_scores
# EVOLVE-BLOCK-END
