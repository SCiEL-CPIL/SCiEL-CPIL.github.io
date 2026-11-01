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

def generate_weight_1_vectors():
    """Generate all weight-1 vectors with entry +/- 2 (norm^2 = 4)"""
    candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def find_cliques(conflicts, max_clique_size=4, max_cliques=50):
    """Find cliques in the conflict graph using a simple heuristic."""
    N = len(conflicts)
    cliques = []
    visited = np.zeros(N, dtype=bool)
    
    for start in range(N):
        if visited[start]:
            continue
        clique = [start]
        visited[start] = True
        
        candidates = np.where(conflicts[start])[0]
        for cand in candidates:
            if visited[cand]:
                continue
            if all(conflicts[cand][c] for c in clique):
                clique.append(cand)
                visited[cand] = True
            if len(clique) >= max_clique_size:
                break
        
        if len(clique) >= 3:
            cliques.append(clique)
        if len(cliques) >= max_cliques:
            break
    
    return cliques

def blp_cutting_plane(conflicts, deadline, rng, initial_solution=None):
    """Solve Maximum Independent Set via BLP with cutting-plane method."""
    N = len(conflicts)
    if N == 0:
        return []
    
    if initial_solution is not None:
        x = np.zeros(N)
        for i in initial_solution:
            x[i] = 1.0
    else:
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
    
    edge_rows = []
    for i in range(N):
        neighbors = np.where(conflicts[i])[0]
        neighbors = neighbors[neighbors > i]
        for j in neighbors:
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            edge_rows.append(row)
    
    if not edge_rows:
        return list(range(N))
    
    A_edges = np.array(edge_rows, dtype=np.float64)
    b_edges = np.ones(len(edge_rows))
    
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
    
    A = np.vstack([A_edges, A_cliques]) if A_cliques.shape[0] > 0 else A_edges
    b = np.concatenate([b_edges, b_cliques]) if A_cliques.shape[0] > 0 else b_edges
    
    c = -np.ones(N)
    bounds = [(0, 1)] * N
    
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if result.status != 0:
        return list(np.where(x > 0.5)[0])
    
    x_lp = result.x
    x_rounded = np.round(x_lp).astype(int)
    
    selected = list(np.where(x_rounded == 1)[0])
    i = 0
    while i < len(selected):
        j = i + 1
        while j < len(selected):
            if conflicts[selected[i]][selected[j]]:
                if x_lp[selected[i]] >= x_lp[selected[j]]:
                    selected.pop(j)
                else:
                    selected.pop(i)
                    break
            j += 1
        i += 1
    
    selected_set = set(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(N):
            if v in selected_set:
                continue
            if not any(conflicts[v][s] for s in selected_set):
                selected_set.add(v)
                improved = True
    
    result_list = list(selected_set)
    
    if len(result_list) > len(best_solution):
        best_solution = np.zeros(N)
        for i in result_list:
            best_solution[i] = 1
        best_obj = len(result_list)
    
    return list(np.where(best_solution > 0.5)[0])

def greedy_mis(conflicts, deadline, rng):
    """Greedy Maximum Independent Set with degree-based ordering."""
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
            
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def kempe_chain_swap(selected, conflicts, deadline, rng):
    """
    Kempe-chain swap: find a vertex outside the set that conflicts with 
    few selected vertices, remove those conflicts, and add the new vertex.
    This can escape local optima of greedy add-only search.
    """
    N = len(conflicts)
    selected_set = set(selected)
    current_list = list(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(current_list)
        
        for idx in range(len(current_list)):
            if time.time() > deadline:
                return current_list
            
            s = current_list[idx]
            
            # Find unselected vertices that conflict with exactly one selected vertex (s)
            # and no other selected vertices
            other_selected = [x for x in current_list if x != s]
            other_set = set(other_selected)
            
            if not other_set:
                # No other selected, just add any unselected non-conflicting vertex
                for v in range(N):
                    if v not in selected_set and not conflicts[v][s]:
                        current_list[idx] = v
                        selected_set.discard(s)
                        selected_set.add(v)
                        improved = True
                        break
                if improved:
                    break
                continue
            
            # For each unselected vertex, count how many selected it conflicts with
            best_swap = None
            best_conflict_count = 2  # Only consider swaps with <= 1 conflict to keep size same or +1
            
            # Sample a subset of unselected vertices for efficiency
            unselected = [v for v in range(N) if v not in selected_set]
            if len(unselected) > 100:
                sample = rng.choice(unselected, size=100, replace=False)
            else:
                sample = unselected
            
            for v in sample:
                # Count conflicts with current selected set
                conflict_count = sum(1 for s in current_list if conflicts[v][s])
                if conflict_count <= 1:
                    # Check if it conflicts with s specifically
                    if conflicts[v][s] and conflict_count == 1:
                        # Swapping s for v keeps size same
                        if best_swap is None or conflict_count < best_conflict_count:
                            best_swap = v
                            best_conflict_count = conflict_count
                    elif conflict_count == 0:
                        # Can just add v (but we're in swap mode, so only if we want to grow)
                        pass
            
            if best_swap is not None:
                current_list[idx] = best_swap
                selected_set.discard(s)
                selected_set.add(best_swap)
                improved = True
                break
    
    return current_list

def local_improve(selected, vectors, dots, conflicts, deadline, rng):
    """Try to improve the selected set by swaps and Kempe chains."""
    N = len(vectors)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected] = True
    current_selected = list(selected)
    
    # First, try to add any non-conflicting vertices
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(N):
            if v in set(current_selected):
                continue
            if not any(conflicts[v][s] for s in current_selected):
                current_selected.append(v)
                selected_mask[v] = True
                improved = True
    
    # Then try Kempe-chain swaps
    if time.time() < deadline - 2:
        current_selected = kempe_chain_swap(current_selected, conflicts, deadline - 1, rng)
    
    # Then try standard swap improvements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(current_selected)
        
        for idx in range(len(current_selected)):
            if time.time() > deadline:
                return current_selected
            
            s_out = current_selected[idx]
            
            other_selected_mask = selected_mask.copy()
            other_selected_mask[s_out] = False
            
            if np.any(other_selected_mask):
                has_conflict = conflicts[:, other_selected_mask].any(axis=1)
            else:
                has_conflict = np.zeros(N, dtype=bool)
                
            candidates = ~has_conflict & ~selected_mask
            
            if np.any(candidates):
                cand_indices = np.where(candidates)[0]
                u_in = cand_indices[rng.integers(len(cand_indices))]
                
                selected_mask[s_out] = False
                selected_mask[u_in] = True
                current_selected[idx] = u_in
                improved = True
                break
    
    return current_selected

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin
    
    # Generate weight-4 vectors
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    # Generate weight-1 vectors (all compatible with weight-4 and each other)
    weight1_vectors = generate_weight_1_vectors()
    
    if N == 0:
        return weight1_vectors
        
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2 (since norm^2 = 4, need dist^2 >= 4, so 4+4-2*dot >= 4 => dot <= 2)
    conflicts = (dots > 2).astype(np.int8)
    
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Get initial solution from greedy
    initial_greedy = greedy_mis(conflicts, deadline - 50, rng)
    best_set = initial_greedy
    
    # Try BLP cutting-plane method
    if time.time() < deadline - 20:
        blp_solution = blp_cutting_plane(conflicts, deadline - 10, rng, initial_solution=initial_greedy)
        if len(blp_solution) > len(best_set):
            best_set = blp_solution
    
    # Run additional greedy restarts
    while time.time() < deadline - 15:
        current_set = greedy_mis(conflicts, deadline - 15, rng)
        if len(current_set) > len(best_set):
            best_set = current_set
        
        if len(best_set) > 0 and time.time() < deadline - 5:
            improved_set = local_improve(list(best_set), vectors, dots, conflicts, deadline - 2, rng)
            if len(improved_set) > len(best_set):
                best_set = improved_set
    
    if not best_set:
        return weight1_vectors
        
    result_vectors_w4 = vectors[best_set].astype(np.int64)
    
    # Combine weight-4 and weight-1 vectors
    final_vectors = np.vstack([result_vectors_w4, weight1_vectors])
    
    return final_vectors
