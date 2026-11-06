"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def priority(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Parse predecessors
    # Assume predecessors is a list of lists or similar structure
    preds = instance.get("predecessors", [])
    if not preds:
        preds = [[] for _ in range(n)]
    
    # Ensure preds are lists of indices
    parsed_preds = []
    for i in range(n):
        p = preds[i]
        if isinstance(p, (list, tuple, np.ndarray)):
            parsed_preds.append(list(p))
        elif isinstance(p, int):
            parsed_preds.append([p])
        else:
            parsed_preds.append([])

    # Build successors for topological sort / eligibility checking
    succs = [[] for _ in range(n)]
    indegree = [0] * n
    for i in range(n):
        for p in parsed_preds[i]:
            succs[p].append(i)
            indegree[i] += 1

    # Topological sort to get a valid order for simulation
    # We need a stable topological order. We'll use Kahn's algorithm.
    # However, for the simulation, we just need to know when a task is eligible.
    # Eligibility: all predecessors are assigned.
    
    # Simulation Strategy:
    # We want to estimate the "slack" of each task.
    # Slack = (Cycle Time - Current Station Remaining Time) - Task Time?
    # Actually, if we pack greedily, a task with time t is placed in the first station 
    # where remaining >= t.
    # If remaining - t is small, it's "tight". If we prioritize tight fits, we fill stations better.
    # But we don't know the order.
    
    # Alternative: Use a heuristic score based on "Critical Path" or "Rank".
    # The current seed is just `times`. 
    # A better heuristic is "Longest Tail" or "Rank" (longest path from task to end).
    # Let's compute the rank (longest path from node to sink).
    # This is a standard heuristic for ALB: Prioritize tasks with the longest downstream path.
    
    # Calculate Rank (Longest path from i to any successor, including i's time)
    # Reverse topological order
    topo_order = []
    in_deg = indegree[:]
    queue = [i for i in range(n) if in_deg[i] == 0]
    # Use a stack or queue. For determinism, sort if needed, but standard topo is fine.
    # We just need the order to compute ranks backwards.
    
    # To ensure we process all nodes (handle cycles gracefully, though task says generated instances, likely DAG)
    # We'll use a simple iterative approach.
    visited = [False] * n
    # Simple DFS for Topological Sort to get reverse order
    def dfs(u):
        if visited[u]:
            return
        visited[u] = True
        for v in succs[u]:
            dfs(v)
        topo_order.append(u)
        
    # Increase recursion limit if n is large, but 30 tasks is small.
    for i in range(n):
        if not visited[i]:
            dfs(i)
            
    # topo_order is in reverse topological order (sinks first)
    # Rank[i] = times[i] + max(Rank[j] for j in succs[i])
    rank = [0.0] * n
    for u in topo_order:
        max_succ_rank = 0.0
        for v in succs[u]:
            if rank[v] > max_succ_rank:
                max_succ_rank = rank[v]
        rank[u] = times[u] + max_succ_rank
        
    # Heuristic: Prioritize by Rank (Longest Tail).
    # This is often better than just time because it accounts for the bottleneck of the line.
    
    # Let's try a combination: Rank + Time? Or just Rank?
    # Let's try Rank first.
    
    # Another strong heuristic: "Weighted Shortest Processing Time" or similar?
    # Actually, for ALB, "Highest Rank" (Longest Path) is a classic heuristic (Jackson's rule).
    
    # Let's verify if Rank is better than Time.
    # If I have Task A (time 10, rank 10) and Task B (time 5, rank 50).
    # B is on the critical path. Placing B early might be better?
    # Actually, usually we want to place tasks that have many successors or long tails early?
    # Wait, if I place a task with a long tail early, it doesn't necessarily help fill the station.
    # The goal is to minimize stations.
    # Greedy: Pick highest priority eligible task that fits.
    # If I pick a task with high rank, I am committing to a long chain.
    
    # Let's try a different angle: "Slack" based on a simulated greedy pass.
    # Simulate a greedy pass using Rank as priority.
    # Record the "remaining time" in the station when each task was placed.
    # Use that remaining time as a proxy for "how tight" the fit was.
    # Tasks that were placed with low remaining time were "critical" to filling that station.
    # But this depends on the order.
    
    # Let's stick to a robust structural heuristic: **Rank (Longest Tail)**.
    # It is computationally cheap (O(N+E)) and theoretically grounded.
    
    # Let's also consider **Successor Count** (out-degree).
    # Tasks with many successors are "bottlenecks".
    
    # Let's try a weighted sum: 0.5 * Rank + 0.5 * Time?
    # Or just Rank.
    
    # Let's check the deadline. We have plenty of time for 30 tasks.
    
    # Let's try to be more sophisticated:
    # Compute the "Position" in the topological sort? No.
    
    # Let's try **Rank** as the primary score.
    # Why? Because tasks with longer tails constrain the schedule more. 
    # Actually, in ALB, the standard heuristics are:
    # 1. Longest Processing Time (LPT) - Current seed
    # 2. Most Immediate Successors (MIS)
    # 3. Longest Tail (Rank)
    # 4. Shortest Tail
    
    # Let's try **Longest Tail (Rank)**.
    
    # One more idea: **Weighted Rank** where weight is time?
    # No, Rank already includes time.
    
    # Compute transitive successor counts (fan-out measure)
    # For each node, count distinct reachable nodes
    trans_count = [0] * n
    for i in range(n):
        visited_set = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for neighbor in succs[node]:
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    stack.append(neighbor)
        trans_count[i] = len(visited_set)
    
    # Normalize rank and trans_count to [0, 1] to combine them effectively
    max_rank = max(rank) if max(rank) > 0 else 1.0
    max_trans = max(trans_count) if max(trans_count) > 0 else 1.0
    
    # Combine: Rank (longest path) + Transitive Count (bottleneck/fan-out)
    # The lesson suggests transitive count is key for unlocking fan-out.
    # We sum normalized values.
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        scores[i] = (rank[i] / max_rank) + (trans_count[i] / max_trans)
        
    return scores

# EVOLVE-BLOCK-END
