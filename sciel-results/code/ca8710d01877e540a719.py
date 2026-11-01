import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2 (i.e., dot >= 3)
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to a more efficient representation for local search
    # Store neighbors as lists for quick access
    neighbor_lists = [np.where(conflicts[i])[0] for i in range(N4)]
    
    best_selected = None
    best_count = 0
    
    def greedy_min_degree(order=None, rng=None):
        """Greedy maximum independent set with minimum degree selection."""
        if rng is None:
            rng = np.random.default_rng(42)
        
        alive = np.ones(N4, dtype=bool)
        selected = []
        degrees = conflicts.sum(axis=1).astype(np.int64).copy()
        
        if order is None:
            # Use minimum degree greedy
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                
                alive_degrees = degrees[alive_indices]
                min_deg = np.min(alive_degrees)
                cands = alive_indices[alive_degrees == min_deg]
                chosen = cands[rng.integers(len(cands))]
                
                selected.append(chosen)
                
                # Remove chosen and its neighbors
                to_remove = np.concatenate(([chosen], neighbor_lists[chosen]))
                to_remove = to_remove[alive[to_remove]]
                alive[to_remove] = False
                
                remaining = np.where(alive)[0]
                if len(remaining) > 0:
                    for r in remaining:
                        # Count how many of r's neighbors were removed
                        removed_neighbors = neighbor_lists[r]
                        removed_count = np.sum(alive[removed_neighbors] == False) 
                        # Actually, we need to count how many of r's neighbors are in to_remove
                        # This is expensive. Let's use a different approach.
                        pass
                    # Simpler: just recompute degrees for remaining
                    # But that's O(N^2). Let's just update incrementally.
                    # For now, skip degree updates for speed and rely on multiple restarts
        else:
            # Use given order
            for i in order:
                if not alive[i]:
                    continue
                selected.append(i)
                to_remove = np.concatenate(([i], neighbor_lists[i]))
                to_remove = to_remove[alive[to_remove]]
                alive[to_remove] = False
        
        return np.array(selected)
    
    def local_search(selected_mask, max_iters=2000):
        """Local search to improve the independent set."""
        selected = selected_mask.copy()
        selected_indices = set(np.where(selected)[0].tolist())
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Try to add any non-selected vertex with no conflicts with selected
            non_selected = np.where(~selected)[0]
            for v in non_selected:
                if time.time() > deadline - 10:
                    break
                # Check if v conflicts with any selected vertex
                conflicts_with_selected = np.any([selected[u] for u in neighbor_lists[v]])
                if not conflicts_with_selected:
                    selected[v] = True
                    selected_indices.add(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one selected, add one non-selected
            # Find a non-selected vertex v that conflicts with exactly one selected vertex u
            # If we remove u and add v, we maintain the same count but might enable further additions
            best_swap = None
            best_gain = 0
            
            for v in non_selected:
                if time.time() > deadline - 10:
                    break
                sel_neighbors = [u for u in neighbor_lists[v] if selected[u]]
                if len(sel_neighbors) == 1:
                    u = sel_neighbors[0]
                    # If we remove u and add v, check if this enables adding more vertices
                    # For now, just do the swap if it doesn't decrease count
                    # Actually, this maintains count. Let's look for swaps that increase count.
                    # A swap increases count if removing u allows adding multiple vertices.
                    # This is complex. Let's try a simpler heuristic.
                    pass
            
            # Try: for each selected vertex, see if removing it allows adding 2+ non-selected vertices
            for u in list(selected_indices):
                if time.time() > deadline - 10:
                    break
                # Temporarily remove u
                selected[u] = False
                selected_indices.discard(u)
                
                # Count how many non-selected vertices can now be added
                addable = 0
                for v in non_selected:
                    if selected[v]:
                        continue
                    # Check if v conflicts with any selected vertex (excluding u)
                    conflicts_with_selected = np.any([selected[w] for w in neighbor_lists[v]])
                    if not conflicts_with_selected:
                        addable += 1
                
                # Restore u
                selected[u] = True
                selected_indices.add(u)
                
                if addable >= 2:
                    # Remove u and add all addable vertices
                    selected[u] = False
                    selected_indices.discard(u)
                    for v in non_selected:
                        if not selected[v]:
                            conflicts_with_selected = np.any([selected[w] for w in neighbor_lists[v]])
                            if not conflicts_with_selected:
                                selected[v] = True
                                selected_indices.add(v)
                    improved = True
                    break
            
            if not improved:
                break
        
        return selected
    
    # Strategy 1: Multiple greedy min-degree restarts with different random seeds
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        rng = np.random.default_rng(1000 + restart)
        selected_arr = greedy_min_degree(rng=rng)
        if len(selected_arr) == 0:
            continue
        
        selected_mask = np.zeros(N4, dtype=bool)
        selected_mask[selected_arr] = True
        
        # Local search refinement
        refined = local_search(selected_mask, max_iters=500)
        count = refined.sum()
        
        if count > best_count:
            best_count = count
            best_selected = refined.copy()
    
    # Strategy 2: Spectral initialization
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    try:
        degree = conflicts.sum(axis=1)
        L = np.diag(degree.astype(np.float64)) - conflicts.astype(np.float64)
        L_sparse = csr_matrix(L)
        
        eigvals, eigvecs = eigsh(L_sparse, k=3, sigma=0, which='LM')
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
        
        # Use Fiedler vector to create an ordering
        order = np.argsort(fiedler)
        
        selected_arr = greedy_min_degree(order=order)
        if len(selected_arr) > 0:
            selected_mask = np.zeros(N4, dtype=bool)
            selected_mask[selected_arr] = True
            refined = local_search(selected_mask, max_iters=1000)
            count = refined.sum()
            if count > best_count:
                best_count = count
                best_selected = refined.copy()
    except Exception:
        pass
    
    # Strategy 3: Random orderings
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        rng = np.random.default_rng(2000 + restart)
        order = rng.permutation(N4)
        selected_arr = greedy_min_degree(order=order)
        if len(selected_arr) > 0:
            selected_mask = np.zeros(N4, dtype=bool)
            selected_mask[selected_arr] = True
            refined = local_search(selected_mask, max_iters=300)
            count = refined.sum()
            if count > best_count:
                best_count = count
                best_selected = refined.copy()
    
    if best_selected is None:
        # Fallback: just use all V1 vectors
        return vectors_v1
    
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors (they're compatible with V4 and each other)
    # Check: V1 vectors have norm 2, V4 vectors have norm 2.
    # Distance between V1 and V4: need dist^2 >= 4.
    # V1 = (2,0,...,0), V4 has four 1s. Dot product can be -2, 0, or 2.
    # If dot = 2, dist^2 = 4 + 4 - 2*2 = 4. OK.
    # If dot = 0, dist^2 = 8. OK.
    # If dot = -2, dist^2 = 12. OK.
    # So all V1 vectors are compatible with all V4 vectors.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
