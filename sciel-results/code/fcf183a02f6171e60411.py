import numpy as np
import itertools
import time
import sys

D = 11

def generate_vectors(weight):
    """Generate all vectors of given weight in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), weight):
        for signs in itertools.product([1, -1], repeat=weight):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set with degree-based ordering.
    """
    N = len(conflicts)
    if N == 0: return []
    
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

def tomata_mis(adjacency, deadline, rng, initial_solution):
    """
    Tomita Maximum Independent Set algorithm.
    """
    N = adjacency.shape[0]
    if N == 0: return initial_solution
    
    best_size = len(initial_solution)
    best_set = list(initial_solution)
    
    # Order vertices by degree descending
    degrees = adjacency.sum(axis=1)
    order = np.argsort(-degrees)
    
    # To speed up, we can use a simple recursion with pruning
    # But for N=5280, recursion depth might be an issue or slow.
    # Let's stick to a robust greedy approach with local search if Tomita is too slow.
    # Actually, Tomita is very effective for sparse graphs.
    
    # We will implement a simplified Tomita:
    # 1. Order vertices.
    # 2. Recursive search with coloring bound.
    
    # For speed, we will use a non-recursive or shallow recursive approach?
    # Let's try a recursive one with a time check.
    
    sys.setrecursionlimit(10000)
    
    # Precompute neighbors as boolean arrays for fast masking
    # adjacency is int8, 1 for conflict
    
    def search(P, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if len(P) == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
            
        # Simple upper bound: len(P)
        if len(current_set) + len(P) <= best_size:
            return
            
        # Pick a vertex u. 
        # Heuristic: Pick vertex with max degree in P?
        # Or just the first one in the ordered P?
        # Tomita uses the coloring order.
        # Let's just pick the first one for simplicity, relying on the initial ordering.
        u = P[0]
        
        # Branch 1: Include u
        # New P = P \ N[u]
        # N[u] are neighbors of u.
        # We need P elements that are NOT neighbors of u and not u.
        
        # mask: 1 if v is NOT neighbor of u and v != u
        # adjacency[u] has 1 for neighbors.
        # ~adjacency[u] has 1 for non-neighbors (including u).
        # So we need to explicitly remove u.
        
        # Efficiently:
        # new_P = P[~adjacency[u][P].astype(bool)]
        # Then remove u.
        
        # To avoid creating large intermediate arrays, we can use:
        # But P is small in deep recursion.
        
        new_P = P[~adjacency[u][P].astype(bool)]
        # Remove u if present
        # Since u is in P, and adj[u][u]=0, u is in new_P.
        # We must remove it.
        # Find index of u in new_P
        # Since P is ordered, and u is P[0], u is likely new_P[0]?
        # Not necessarily, if P[0] is u, then new_P[0] is u.
        # So we can just do new_P = new_P[1:] if u == new_P[0]?
        # Wait, P[0] is u.
        # new_P is derived from P.
        # If u is in new_P, it's because adj[u][u]=0.
        # Is u the first element of new_P?
        # P is sorted by degree. u is P[0].
        # new_P is a subset of P.
        # So u is the smallest element in new_P (if P is sorted by index? No, by degree).
        # But we can just filter.
        
        # Faster:
        # We know u is in new_P.
        # Let's just remove the first occurrence.
        # Or better, construct new_P excluding u from the start?
        # P[1:] are the others.
        # Check which of P[1:] are not neighbors of u.
        
        others = P[1:]
        if len(others) > 0:
             mask = ~adjacency[u][others].astype(bool)
             new_P = others[mask]
        else:
             new_P = np.array([], dtype=int)
             
        search(new_P, current_set + [u])
        
        # Branch 2: Exclude u
        search(P[1:], current_set)
        
    # Initial P is all vertices
    P = np.arange(N)
    search(P, [])
    
    return best_set

def construct():
    t0 = time.time()
    deadline = t0 + 115
    
    rng = np.random.default_rng(42)
    
    # 1. Generate W4 vectors
    w4_vectors = generate_vectors(4)
    N4 = len(w4_vectors)
    
    # 2. Build conflict graph for W4
    # Conflict if dot product >= 3
    dots4 = w4_vectors @ w4_vectors.T
    conflicts4 = (dots4 >= 3).astype(np.int8)
    np.fill_diagonal(conflicts4, 0)
    
    # 3. Solve MIS on W4
    # Use Tomita if time permits, else Greedy
    # Tomita might be slow for 5280 nodes if not optimized.
    # Let's try Greedy with multiple restarts first, it's faster.
    # Then try Tomita on the result?
    
    best_set4 = []
    
    # Multiple greedy restarts
    for i in range(20):
        if time.time() > deadline - 30: break
        cur_set = greedy_mis(conflicts4, deadline - 30, rng)
        if len(cur_set) > len(best_set4):
            best_set4 = cur_set
            
    # Try Tomita with remaining time
    if time.time() < deadline - 10:
        tomata_deadline = deadline - 2
        improved_set4 = tomata_mis(conflicts4, tomata_deadline, rng, best_set4)
        if len(improved_set4) > len(best_set4):
            best_set4 = improved_set4
            
    selected_w4 = w4_vectors[best_set4]
    
    # 4. Add W3 vectors
    w3_vectors = generate_vectors(3)
    N3 = len(w3_vectors)
    
    # Check compatibility with selected W4
    # Condition: u (W3) . v (W4) <= 1
    if len(selected_w4) > 0:
        dots34 = w3_vectors @ selected_w4.T
        # A W3 vector is compatible if max dot product with any selected W4 is <= 1
        # i.e., no dot product >= 2
        compatible3 = (dots34 < 2).all(axis=1)
    else:
        compatible3 = np.ones(N3, dtype=bool)
        
    w3_candidates = w3_vectors[compatible3]
    N3c = len(w3_candidates)
    
    if N3c > 0:
        # Build conflict graph among W3 candidates
        # Condition: u (W3) . v (W3) <= 1
        dots33 = w3_candidates @ w3_candidates.T
        conflicts33 = (dots33 >= 2).astype(np.int8)
        np.fill_diagonal(conflicts33, 0)
        
        best_set3 = []
        for i in range(10):
            if time.time() > deadline - 10: break
            cur_set = greedy_mis(conflicts33, deadline - 10, rng)
            if len(cur_set) > len(best_set3):
                best_set3 = cur_set
                
        selected_w3 = w3_candidates[best_set3]
    else:
        selected_w3 = np.empty((0, D), dtype=np.int8)
        
    # 5. Add W2 vectors
    w2_vectors = generate_vectors(2)
    N2 = len(w2_vectors)
    
    # Check compatibility with selected W4 and W3
    # Condition with W4: u (W2) . v (W4) <= 1
    # Condition with W3: u (W2) . v (W3) <= 1
    if len(selected_w4) > 0:
        dots24 = w2_vectors @ selected_w4.T
        comp24 = (dots24 < 2).all(axis=1)
    else:
        comp24 = np.ones(N2, dtype=bool)
        
    if len(selected_w3) > 0:
        dots23 = w2_vectors @ selected_w3.T
        comp23 = (dots23 < 2).all(axis=1)
    else:
        comp23 = np.ones(N2, dtype=bool)
        
    compatible2 = comp24 & comp23
    w2_candidates = w2_vectors[compatible2]
    N2c = len(w2_candidates)
    
    if N2c > 0:
        # Build conflict graph among W2 candidates
        # Condition: u (W2) . v (W2) <= 0
        dots22 = w2_candidates @ w2_candidates.T
        conflicts22 = (dots22 >= 1).astype(np.int8)
        np.fill_diagonal(conflicts22, 0)
        
        best_set2 = []
        for i in range(10):
            if time.time() > deadline - 2: break
            cur_set = greedy_mis(conflicts22, deadline - 2, rng)
            if len(cur_set) > len(best_set2):
                best_set2 = cur_set
                
        selected_w2 = w2_candidates[best_set2]
    else:
        selected_w2 = np.empty((0, D), dtype=np.int8)
        
    # Combine results
    all_vectors = np.vstack([selected_w4, selected_w3, selected_w2])
    
    if len(all_vectors) == 0:
        # Fallback to a single vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return all_vectors.astype(np.int64)
