import time
import math
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # Safety margin before 60s kill

    N = 256
    
    # 1. Precompute squares and forbidden nodes
    # A node 'a' is forbidden if 2*a is a perfect square (diagonal constraint a+a is square)
    # Also, we need to know which pairs (a,b) conflict (a+b is square)
    
    max_sum = 2 * N
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
        
    # Identify valid nodes (those where 2*a is NOT a square)
    # If 2*a is a square, x_a must be 0.
    valid_nodes = []
    for i in range(1, N + 1):
        if 2 * i not in squares:
            valid_nodes.append(i)
            
    M = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # If no valid nodes (unlikely), return empty or single
    if M == 0:
        return []
        
    # 2. Build Conflict Matrix for ILP
    # We want to maximize sum(x_i)
    # Subject to x_a + x_b <= 1 for all pairs (a,b) in valid_nodes where a+b is square.
    # Note: We only consider pairs from valid_nodes.
    # If a node is invalid (forbidden), it's not in the ILP variables, effectively x=0.
    
    # We will build the constraint matrix A for scipy.optimize.milp
    # Variables: x_0 ... x_{M-1} corresponding to valid_nodes[0] ... valid_nodes[M-1]
    
    rows = []
    # Precompute conflicts for each node to speed up row construction
    # conflict_pairs[i] is a list of indices j such that valid_nodes[i] + valid_nodes[j] is square
    # Since the graph is undirected, we can just iterate i < j
    
    conflict_pairs = [[] for _ in range(M)]
    
    for i in range(M):
        a = valid_nodes[i]
        for j in range(i + 1, M):
            b = valid_nodes[j]
            if a + b in squares:
                conflict_pairs[i].append(j)
                conflict_pairs[j].append(i)
                
    # Build rows for constraints: x_i + x_j <= 1
    for i in range(M):
        for j in conflict_pairs[i]:
            if j > i: # Avoid duplicate rows
                row = np.zeros(M)
                row[i] = 1.0
                row[j] = 1.0
                rows.append(row)
                
    if not rows:
        # No conflicts, take all
        return valid_nodes
        
    A = np.array(rows)
    b_ub = np.ones(len(rows))
    
    # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
    c = -np.ones(M)
    
    # Bounds: 0 <= x_i <= 1
    bounds = (0, 1)
    
    # Integrality: All binary
    integrality = np.ones(M)
    
    # Try to solve with MILP
    solution_found = False
    best_set = []
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraints = LinearConstraint(A, ub=b_ub)
        bounds_obj = Bounds(lb=0, ub=1)
        
        # Timeout for milp
        # scipy milp doesn't have a direct timeout parameter in older versions,
        # but we can rely on the global time check if we run it in a way that allows interruption?
        # Actually, milp is a C++ call. We should be careful.
        # However, for N=256, this instance is usually fast for HiGHS.
        
        res = milp(c=c, constraints=constraints, bounds=bounds_obj, integrality=integrality)
        
        if res.status == 0:
            # res.x is the optimal solution
            selected_indices = [i for i in range(M) if res.x[i] > 0.5]
            best_set = [valid_nodes[i] for i in selected_indices]
            solution_found = True
        elif res.status == 1:
            # "Optimal solution found (within tolerances)"
            selected_indices = [i for i in range(M) if res.x[i] > 0.5]
            best_set = [valid_nodes[i] for i in selected_indices]
            solution_found = True
            
    except Exception as e:
        pass
        
    if solution_found:
        # Verify the solution
        # Check if any pair in best_set sums to a square
        valid = True
        s_set = set(best_set)
        for i in range(len(best_set)):
            a = best_set[i]
            for j in range(i, len(best_set)):
                b = best_set[j]
                if a + b in squares:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return best_set
        else:
            # If ILP returned an invalid solution (shouldn't happen, but safe to check)
            pass

    # Fallback: Greedy / Local Search if MILP fails or times out
    # This is a fast heuristic to ensure we return *something* good
    if time.time() - start_time > TIME_LIMIT - 5:
        # Very tight time, return a quick greedy
        current = set()
        # Sort by degree (fewest conflicts first) to pack more? Or most conflicts?
        # Usually, picking nodes with fewer conflicts first is safer for independence.
        # Let's just do a simple greedy
        for v in valid_nodes:
            idx = node_to_idx[v]
            # Check conflicts
            conflict = False
            for neighbor in conflict_pairs[idx]:
                if valid_nodes[neighbor] in current:
                    conflict = True
                    break
            if not conflict:
                current.add(v)
        return list(current)

    # If we have time, we could try a simple branch and bound or local search improvement
    # But given the directive to keep the design and use the ILP, and ILP usually works,
    # we return the greedy result if MILP failed.
    
    # Let's try a slightly better greedy: sort by degree ascending
    degrees = [len(conflict_pairs[i]) for i in range(M)]
    sorted_indices = sorted(range(M), key=lambda i: degrees[i])
    
    current_set_mask = 0
    current_set = set()
    for idx in sorted_indices:
        v = valid_nodes[idx]
        # Check if v conflicts with any in current_set
        # We can use bitmasks for speed
        # But simple set check is fine for 256 nodes
        conflict = False
        for neighbor in conflict_pairs[idx]:
            if valid_nodes[neighbor] in current_set:
                conflict = True
                break
        if not conflict:
            current_set.add(v)
            
    return list(current_set)
