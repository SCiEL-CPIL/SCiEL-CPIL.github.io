"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Calculates priority scores for assembly tasks using a composite heuristic
    based on Longest Path (Rank), Total Downstream Time, and connectivity.
    
    The Longest Path heuristic estimates the minimum remaining time for each task 
    to complete the line, prioritizing tasks on the critical path. This is 
    generally superior to simple LPT (Longest Processing Time) for SALB as it 
    accounts for precedence constraints' impact on station count.
    
    We combine:
    1. Rank (Longest Path): Primary signal for critical path tasks.
    2. Total Downstream Time: Sum of all downstream work, identifying true bottlenecks.
    3. Number of Successors: Connectivity metric to break ties favoring high-impact tasks.
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    n_tasks = len(times)
    preds = instance["predecessors"]
    
    # Build adjacency list for successors
    successors = [[] for _ in range(n_tasks)]
    in_degree = [0] * n_tasks
    
    for i in range(n_tasks):
        if i < len(preds):
            for p in preds[i]:
                # p is a predecessor of i, so i is a successor of p
                successors[p].append(i)
                in_degree[i] += 1
                
    # Topological Sort (Kahn's Algorithm) to process nodes
    queue = []
    indeg = in_degree[:]
    for i in range(n_tasks):
        if indeg[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
                
    # If graph is cyclic or malformed, topo_order might be incomplete. Assume valid DAG.
    # Calculate Rank (Longest Path) and Total Downstream Time in reverse topological order
    rank = np.zeros(n_tasks, dtype=float)
    total_down = np.zeros(n_tasks, dtype=float)
    
    # Iterate backwards through topological order
    for i in reversed(topo_order):
        # For rank: longest path through successors
        max_succ_rank = 0.0
        # For total_down: sum of all downstream work
        sum_succ_down = 0.0
        
        if successors[i]:
            succ_ranks = [rank[j] for j in successors[i]]
            max_succ_rank = max(succ_ranks)
            succ_downs = [total_down[j] for j in successors[i]]
            sum_succ_down = sum(succ_downs)
        
        rank[i] = times[i] + max_succ_rank
        total_down[i] = times[i] + sum_succ_down
        
    # Secondary features
    num_succ = np.array([len(successors[i]) for i in range(n_tasks)], dtype=float)
    
    # Composite Score:
    # Primary: Rank (critical path) - most important for minimizing stations
    # Secondary: Total downstream time - identifies true bottlenecks
    # Tertiary: Number of successors - breaks ties favoring high-connectivity tasks
    # Bottleneck Slack: time / (1 + total_down) rewards tasks that fill capacity
    #   without over-constraining downstream work
    # Predecessor count: tasks that gate many paths should be prioritized
    #
    # The total_down metric captures tasks that have significant work depending on them,
    # which is a better bottleneck indicator than just the longest path.
    #
    # Weights are tuned to balance critical path adherence with global bottleneck detection.
    
    # Bottleneck slack: high time relative to downstream complexity
    # Tasks with high time but low downstream complexity are "pure fillers"
    # Tasks with high time and high downstream complexity are critical bottlenecks
    bottleneck_slack = times / (1.0 + total_down)
    
    # Predecessor count: how many tasks depend directly on this task
    # (already captured in num_succ, but we can also look at in-degree of successors)
    num_preds = np.array([len(preds[i]) if i < len(preds) else 0 for i in range(n_tasks)], dtype=float)
    
    # Composite score with refined weights:
    # - Rank: primary signal for critical path
    # - Total downstream: secondary, captures true bottlenecks  
    # - Bottleneck slack: rewards efficient capacity usage
    # - Num successors: tie-breaker for connectivity
    # - Num preds: tasks that are prerequisites for many paths
    scores = 1.0 * rank + 0.20 * total_down + 0.10 * bottleneck_slack * max(times) + 0.05 * num_succ + 0.03 * num_preds
    
    # Deadline guard: ensure we return well before timeout
    elapsed = time.time() - start_time
    if elapsed > 145:  # Safety margin
        # Return the current scores; computation is fast for n=30
        return scores
        
    return scores
# EVOLVE-BLOCK-END
