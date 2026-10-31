import numpy as np
import itertools
import time
import sys
from scipy.optimize import linprog

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def find_cliques(conflicts, max_clique_size=4, max_cliques=50):
    """
    Find cliques in the conflict graph using a simple heuristic.
    Returns list of cliques, each clique is a list of node indices.
    """
    N = len(conflicts)
    cliques = []
    visited = np.zeros(N, dtype=bool)
    
    for start in range(N):
        if visited[start]:
            continue
        # Try to build a clique starting from this node
        clique = [start]
        visited[start] = True
        
        # Greedily add nodes that connect to all current clique members
        candidates = np.where(conflicts[start])[0]
        for cand in candidates:
            if visited[cand]:
                continue
            # Check if cand connects to all current clique members
            if all(conflicts[cand][c] for c in clique):
                clique.append(cand)
                visited[cand] = True
            if len(clique) >= max_clique_size:
                break
        
        if len(clique) >= 3:  # Only keep cliques of size >= 3
            cliques.append(clique)
        if len(cliques) >= max_cliques:
            break
    
    return cliques

def blp_cutting_plane(conflicts, deadline, rng, initial_solution=None):
    """
    Solve Maximum Independent Set via BLP with cutting-plane method.
    Variables x_i in {0,1} for each vertex.
    Maximize sum(x_i) subject to x_i + x_j <= 1 for all edges (i,j).
    """
    N = len(conflicts)
    if N == 0:
        return []
    
    # Start with a solution if provided, else use greedy
    if initial_solution is not None:
        x = np.zeros(N)
        for i in initial_solution:
            x[i] = 1.0
    else:
        # Initialize with a simple greedy solution
        x = np.zeros(N)
        alive = np.ones(N, dtype=bool)
        while np.any(alive):
            if time.time() > deadline:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(conflicts[alive_indices].sum(axis=1))
            candidates = alive_indices[conflicts[alive_indices].sum(axis=1) == min_deg]
            chosen = candidates[rng.integers(len(candidates))]
            x[chosen] = 1.0
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
    
    best_solution = x.copy()
    best_obj = x.sum()
    
    # Build constraint matrix for edges
    edge_rows = []
    for i in range(N):
        neighbors = np.where(conflicts[i])[0]
        neighbors = neighbors[neighbors > i]  # Avoid duplicate edges
        for j in neighbors:
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            edge_rows.append(row)
    
    if not edge_rows:
        # No conflicts, select all
        return list(range(N))
    
    A_edges = np.array(edge_rows, dtype=np.float64)
    b_edges = np.ones(len(edge_rows))
    
    # Add clique constraints
    cliques = find_cliques(conflicts, max_clique_size=4, max_cliques=30)
    A_cliques = []
    b_cliques = []
    for clique in cliques:
        row = np.zeros(N)
        for i in clique:
            row[i] = 1
        A_cliques.append(row)
        b_cliques.append(1)
    
    if A_cliques:
        A_cliques = np.array(A_cliques, dtype=np.float64)
        b_cliques = np.array(b_cliques)
    else:
        A_cliques = np.empty((0, N))
        b_cliques = np.empty(0)
    
    # Combined constraints
    A = np.vstack([A_edges, A_cliques]) if A_cliques.shape[0] > 0 else A_edges
    b = np.concatenate([b_edges, b_cliques]) if A_cliques.shape[0] > 0 else b_edges
    
    # Objective: maximize sum(x) => minimize -sum(x)
    c = -np.ones(N)
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1)] * N
    
    # LP relaxation
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if result.status != 0:
        return list(np.where(x > 0.5)[0])
    
    x_lp = result.x
    
    # Round the LP solution
    x_rounded = np.round(x_lp).astype(int)
    
    # Ensure feasibility: remove conflicts
    selected = list(np.where(x_rounded == 1)[0])
    # Remove any conflicting pairs
    i = 0
    while i < len(selected):
        j = i + 1
        while j < len(selected):
            if conflicts[selected[i]][selected[j]]:
                # Remove the one with lower LP value
                if x_lp[selected[i]] >= x_lp[selected[j]]:
                    selected.pop(j)
                else:
                    selected.pop(i)
                    break
            j += 1
        i += 1
    
    # Local improvement: try to add more vertices
    selected_set = set(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(N):
            if v in selected_set:
                continue
            # Check if v conflicts with any selected vertex
            if not any(conflicts[v][s] for s in selected_set):
                selected_set.add(v)
                improved = True
    
    result_list = list(selected_set)
    
    # If this solution is better, update
    if len(result_list) > len(best_solution):
        best_solution = np.zeros(N)
        for i in result_list:
            best_solution[i] = 1
        best_obj = len(result_list)
    
    return list(np.where(best_solution > 0.5)[0])

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set with degree-based ordering.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def local_improve(selected, vectors, dots, conflicts, deadline, rng):
    """
    Try to improve the selected set by swapping out one vector for another.
    Optimized with numpy boolean masking.
    """
    N = len(vectors)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected] = True
    current_selected = list(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle selected indices to avoid bias
        rng.shuffle(current_selected)
        
        for idx in range(len(current_selected)):
            if time.time() > deadline:
                return current_selected
            
            s_out = current_selected[idx]
            
            # Create mask of "other selected"
            other_selected_mask = selected_mask.copy()
            other_selected_mask[s_out] = False
            
            # Find unselected vectors that do not conflict with any other selected
            if np.any(other_selected_mask):
                has_conflict = conflicts[:, other_selected_mask].any(axis=1)
            else:
                has_conflict = np.zeros(N, dtype=bool)
                
            # Candidates are those without conflict, not currently selected
            candidates = ~has_conflict & ~selected_mask
            
            # We need to pick a candidate u to replace s_out.
            if np.any(candidates):
                cand_indices = np.where(candidates)[0]
                u_in = cand_indices[rng.integers(len(cand_indices))]
                
                # Perform swap
                selected_mask[s_out] = False
                selected_mask[u_in] = True
                
                # Update list
                current_selected[idx] = u_in
                
                improved = True
                break # Restart the loop with the new configuration
    
    return current_selected

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin
    
    # Generate weight-4 vectors
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2 (since norm^2 = 4, need dist^2 >= 4, so 4+4-2*dot >= 4 => dot <= 2)
    conflicts = (dots > 2).astype(np.int8)
    
    # Self-conflicts don't matter since we pick distinct vectors
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # First, get a good initial solution from greedy
    initial_greedy = greedy_mis(conflicts, deadline - 50, rng)
    best_set = initial_greedy
    
    # Try BLP cutting-plane method
    if time.time() < deadline - 20:
        blp_solution = blp_cutting_plane(conflicts, deadline - 10, rng, initial_solution=initial_greedy)
        if len(blp_solution) > len(best_set):
            best_set = blp_solution
    
    # Run additional greedy restarts with the best solution as reference
    while time.time() < deadline - 15:
        current_set = greedy_mis(conflicts, deadline - 15, rng)
        if len(current_set) > len(best_set):
            best_set = current_set
        
        # Local improvement
        if len(best_set) > 0 and time.time() < deadline - 5:
            improved_set = local_improve(list(best_set), vectors, dots, conflicts, deadline - 2, rng)
            if len(improved_set) > len(best_set):
                best_set = improved_set
    
    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)
