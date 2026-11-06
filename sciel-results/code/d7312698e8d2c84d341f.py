"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task using a resource-contention metric:
    priority = sum of cycle-time-normalized durations of all transitive predecessors
               + sum of cycle-time-normalized durations of all transitive successors
               + small bonus for number of transitive successors (to break ties favoring bottleneck tasks)
    
    This prioritizes tasks that are part of the most time-dense, interconnected subgraphs,
    maximizing parallelism potential in the greedy station-filling heuristic.
    """
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Build adjacency lists for predecessors and successors
    # predecessors[i] = list of tasks that must come before task i
    # successors[i] = list of tasks that must come after task i
    preds = instance.get("predecessors", None)
    
    # Handle different possible formats for predecessors
    if preds is None:
        pred_list = [[] for _ in range(n)]
        succ_list = [[] for _ in range(n)]
    else:
        if isinstance(preds, (list, tuple)):
            if len(preds) == n:
                # Format: preds[i] = list of predecessor indices for task i
                pred_list = [list(p) for p in preds]
            else:
                # Maybe it's a list of (pred, succ) pairs or similar
                pred_list = [[] for _ in range(n)]
                succ_list = [[] for _ in range(n)]
                for item in preds:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        p, s = int(item[0]), int(item[1])
                        if 0 <= p < n and 0 <= s < n:
                            pred_list[s].append(p)
                            succ_list[p].append(s)
        elif isinstance(preds, dict):
            pred_list = [[] for _ in range(n)]
            succ_list = [[] for _ in range(n)]
            for s, p_list in preds.items():
                s = int(s)
                if 0 <= s < n:
                    pred_list[s] = [int(p) for p in p_list if 0 <= int(p) < n]
                    for p in pred_list[s]:
                        if p not in succ_list[p]:
                            succ_list[p].append(s)
        else:
            pred_list = [[] for _ in range(n)]
            succ_list = [[] for _ in range(n)]
    
    # Ensure succ_list is built if not already
    if preds is not None and not (isinstance(preds, dict)):
        # Rebuild succ_list from pred_list if needed
        succ_list = [[] for _ in range(n)]
        for i in range(n):
            for p in pred_list[i]:
                if i not in succ_list[p]:
                    succ_list[p].append(i)
    
    # Compute transitive predecessors and successors using BFS/DFS
    # trans_preds[i] = set of all tasks that are transitive predecessors of i
    # trans_succs[i] = set of all tasks that are transitive successors of i
    
    def compute_transitive(adj_list):
        """Compute transitive closure for each node."""
        trans = [set() for _ in range(n)]
        for i in range(n):
            if i in trans:  # already computed
                continue
            visited = set()
            stack = [i]
            while stack:
                node = stack.pop()
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            trans[i] = visited
        return trans
    
    # Transitive predecessors: for each task i, follow pred_list backwards
    # We need: for task i, all tasks j such that j must come before i (transitively)
    # This is the reverse graph traversal on pred_list
    trans_preds = [set() for _ in range(n)]
    for i in range(n):
        visited = set()
        stack = list(pred_list[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(pred_list[node])
        trans_preds[i] = visited
    
    # Transitive successors: for each task i, follow succ_list forwards
    trans_succs = [set() for _ in range(n)]
    for i in range(n):
        visited = set()
        stack = list(succ_list[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(succ_list[node])
        trans_succs[i] = visited
    
    # Check deadline
    if time.time() > DEADLINE:
        return times.copy()
    
    # Compute resource contention metric
    # For each task i:
    #   pred_weight = sum of times[j] / cycle_time for j in trans_preds[i]
    #   succ_weight = sum of times[j] / cycle_time for j in trans_succs[i]
    #   successor_count = len(trans_succs[i])
    #   priority[i] = pred_weight + succ_weight + small_bonus * successor_count
    
    normalized_times = times / cycle_time
    
    priorities = np.zeros(n, dtype=float)
    
    for i in range(n):
        # Sum of normalized times of transitive predecessors
        pred_weight = 0.0
        for j in trans_preds[i]:
            pred_weight += normalized_times[j]
        
        # Sum of normalized times of transitive successors
        succ_weight = 0.0
        for j in trans_succs[i]:
            succ_weight += normalized_times[j]
        
        # Small bonus for number of transitive successors (bottleneck/fan-out)
        succ_count = len(trans_succs[i])
        small_bonus = 0.01 * succ_count
        
        priorities[i] = pred_weight + succ_weight + small_bonus
    
    # Also add the task's own duration as a base component
    # This ensures longer tasks still have higher priority, but modulated by contention
    priorities += 0.5 * times
    
    # Check deadline
    if time.time() > DEADLINE:
        return priorities
    
    return priorities
# EVOLVE-BLOCK-END
