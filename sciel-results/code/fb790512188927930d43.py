import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety

    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i*i <= 2*N:
        squares.add(i*i)
        i += 1
    
    # Identify valid nodes (no self-conflict)
    # If 2*v is a square, v cannot be in the set (diagonal constraint a=b)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2*v not in squares:
            valid_nodes.append(v)
            
    n_vars = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict matrix
    # conflict_matrix[i][j] = 1 if valid_nodes[i] + valid_nodes[j] is a square
    # We only need upper triangle for constraints x_i + x_j <= 1
    conflict_pairs = []
    for i in range(n_vars):
        v = valid_nodes[i]
        for j in range(i + 1, n_vars):
            u = valid_nodes[j]
            if v + u in squares:
                conflict_pairs.append((i, j))
                
    # If no conflicts, we can take all valid nodes
    if not conflict_pairs:
        return valid_nodes

    # Build adjacency list for local search and verification
    adj = [set() for _ in range(n_vars)]
    for i, j in conflict_pairs:
        adj[i].add(j)
        adj[j].add(i)

    # Prepare ILP
    c = -np.ones(n_vars)
    num_constraints = len(conflict_pairs)
    A = np.zeros((num_constraints, n_vars))
    b = np.ones(num_constraints)
    
    for k, (i, j) in enumerate(conflict_pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0
        
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars)
    
    best_solution = None
    best_size = 0

    # Try MILP with time guard
    if time.time() < deadline - 10:
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, ub=b),
                integrality=integrality,
                bounds=bounds
            )
            
            if res.success and res.x is not None:
                solution = res.x
                selected_indices = [i for i in range(n_vars) if solution[i] > 0.5]
                # Verify using adjacency list
                sel_set = set(selected_indices)
                is_valid = all(j not in sel_set for i in selected_indices for j in adj[i])
                if is_valid and len(selected_indices) > best_size:
                    best_solution = selected_indices
                    best_size = len(selected_indices)
        except Exception:
            pass

    if best_solution is not None and best_size > 0:
        return [valid_nodes[i] for i in best_solution]

    # Fallback: Randomized greedy + local search with 1->2 swaps
    if time.time() < deadline - 5:
        best_greedy = None
        best_greedy_size = 0
        
        while time.time() < deadline - 5:
            order = list(range(n_vars))
            random.shuffle(order)
            
            selected = set()
            for i in order:
                if all(j not in selected for j in adj[i]):
                    selected.add(i)
            
            # Local search: 1-removal + 2-addition swaps
            improved = True
            while improved:
                improved = False
                current_size = len(selected)
                
                for rem in list(selected):
                    if time.time() > deadline - 5:
                        break
                    new_sel = selected - {rem}
                    # Find all nodes that can be added to new_sel
                    candidates = []
                    for i in range(n_vars):
                        if i not in new_sel and all(j not in new_sel for j in adj[i]):
                            candidates.append(i)
                    
                    # Try adding pairs
                    for a in range(len(candidates)):
                        ca = candidates[a]
                        for b in range(a + 1, len(candidates)):
                            cb = candidates[b]
                            if cb not in adj[ca]:
                                new_size = len(new_sel) + 2
                                if new_size > current_size:
                                    selected = new_sel | {ca, cb}
                                    current_size = new_size
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            if len(selected) > best_greedy_size:
                best_greedy_size = len(selected)
                best_greedy = set(selected)
        
        if best_greedy is not None and len(best_greedy) > 0:
            return [valid_nodes[i] for i in best_greedy]

    # Absolute fallback
    seed_set = [v for v in valid_nodes if v % 3 == 1]
    valid_seed = []
    s_set = set()
    for v in seed_set:
        ok = True
        if 2*v in squares:
            ok = False
        if ok:
            for u in s_set:
                if v + u in squares:
                    ok = False
                    break
        if ok:
            valid_seed.append(v)
            s_set.add(v)
            
    if valid_seed:
        return valid_seed
    else:
        return [valid_nodes[0]] if valid_nodes else [1]
