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
    
    # Build conflict graph
    # conflict_pairs[i][j] = 1 if valid_nodes[i] + valid_nodes[j] is a square
    # We only need upper triangle for constraints x_i + x_j <= 1
    conflict_pairs = []
    # Also build adjacency list for dominance check
    adj = [set() for _ in range(n_vars)]
    
    for i in range(n_vars):
        v = valid_nodes[i]
        for j in range(i + 1, n_vars):
            u = valid_nodes[j]
            if v + u in squares:
                conflict_pairs.append((i, j))
                adj[i].add(j)
                adj[j].add(i)
                
    # If no conflicts, we can take all valid nodes
    if not conflict_pairs:
        return valid_nodes

    # Dominance Reduction:
    # If N[v] is a subset of N[u] (where N[v] is the set of neighbors of v), 
    # then u can be removed from the graph.
    # Why? Any independent set containing u can be transformed to one containing v 
    # (replace u with v) without violating independence, because v's neighbors 
    # are a subset of u's neighbors, so v doesn't conflict with any other node 
    # in the set that u didn't.
    # Note: This reduction is safe for finding *a* maximum independent set, 
    # but we must be careful if we want to map back. 
    # Actually, if we remove u, we assume v is "better". 
    # We keep v, remove u.
    
    to_remove = [False] * n_vars
    # Optimization: Sort by degree descending? 
    # If deg(u) > deg(v), u is less likely to be subset of v? 
    # If N[v] subset N[u], then deg(v) <= deg(u).
    # So we only need to check pairs where deg(v) <= deg(u).
    
    degrees = [len(adj[i]) for i in range(n_vars)]
    # Sort indices by degree
    sorted_indices = sorted(range(n_vars), key=lambda x: degrees[x])
    
    # For each node u (higher degree), check if there is a node v (lower/equal degree) 
    # such that adj[v] is subset of adj[u].
    # If so, remove u.
    
    # This is O(N^2 * D) roughly. N=245, D is small.
    # 245^2 * 10 ~ 600,000 ops. Very fast.
    
    for i_idx, u in enumerate(sorted_indices):
        if to_remove[u]:
            continue
        # Check all v with degree <= degree[u]
        # Since sorted_indices is sorted by degree, we can just check previous ones?
        # No, v can be any node with smaller degree.
        for j_idx in range(i_idx):
            v = sorted_indices[j_idx]
            if to_remove[v]:
                continue
            # If deg(v) > deg(u), break? No, sorted ascending, so deg(v) <= deg(u).
            # Check if adj[v] is subset of adj[u]
            # adj[v] and adj[u] are sets of indices.
            if adj[v].issubset(adj[u]):
                to_remove[u] = True
                break

    # Create new list of active nodes
    active_indices = [i for i in range(n_vars) if not to_remove[i]]
    active_nodes = [valid_nodes[i] for i in active_indices]
    
    # Map old index to new index
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(active_indices)}
    
    # Rebuild conflict pairs for the reduced graph
    reduced_conflict_pairs = []
    for i, j in conflict_pairs:
        if i in old_to_new and j in old_to_new:
            reduced_conflict_pairs.append((old_to_new[i], old_to_new[j]))
            
    n_active = len(active_indices)
    
    if n_active == 0:
        return valid_nodes # Should not happen

    if not reduced_conflict_pairs:
        return active_nodes

    # Prepare ILP
    # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
    c = -np.ones(n_active)
    
    # Constraints: x_i + x_j <= 1 for all conflicts
    num_constraints = len(reduced_conflict_pairs)
    A = np.zeros((num_constraints, n_active))
    b = np.ones(num_constraints)
    
    for k, (i, j) in enumerate(reduced_conflict_pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0
        
    # Bounds: x_i in [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(n_active)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, ub=b),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            # Extract solution
            solution = res.x
            # Since it's binary, values are close to 0 or 1
            selected_active_indices = [i for i in range(n_active) if solution[i] > 0.5]
            
            # Map back to original valid_nodes indices
            selected_original_indices = [active_indices[i] for i in selected_active_indices]
            selected_nodes = [valid_nodes[i] for i in selected_original_indices]
            
            # Sanity check: Verify validity
            is_valid = True
            selected_set = set(selected_nodes)
            for v in selected_nodes:
                if 2*v in squares:
                    is_valid = False
                    break
            if is_valid:
                for i, v in enumerate(selected_nodes):
                    for u in selected_nodes[i+1:]:
                        if v + u in squares:
                            is_valid = False
                            break
                    if not is_valid:
                        break
            
            if is_valid and len(selected_nodes) > 0:
                return selected_nodes
            elif is_valid:
                # Fallback if ILP returns empty (shouldn't happen)
                pass
            
        # Fallback if ILP fails or returns invalid
    except Exception as e:
        pass

    # Fallback: Simple Greedy MIS if ILP fails or times out
    # Use the original valid_nodes
    # Greedy: Pick node with lowest degree first? Or highest?
    # For MIS, picking low degree nodes is often better as they block fewer others.
    
    # Calculate degrees in original graph
    orig_degrees = [len(adj[i]) for i in range(n_vars)]
    greedy_order = sorted(range(n_vars), key=lambda x: orig_degrees[x])
    
    selected = []
    selected_set = set()
    
    for idx in greedy_order:
        v = valid_nodes[idx]
        # Check if v conflicts with any selected node
        ok = True
        if 2*v in squares:
            ok = False
        if ok:
            for u in selected:
                if v + u in squares:
                    ok = False
                    break
        if ok:
            selected.append(v)
            selected_set.add(v)
            
    if selected:
        return selected
    else:
        # Absolute fallback: pick first valid node
        return [valid_nodes[0]] if valid_nodes else [1]
