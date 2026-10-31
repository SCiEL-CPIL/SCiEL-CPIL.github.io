import numpy as np
import itertools
import time
import sys

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
    
    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert conflicts to int for efficient operations
    conflicts_int = conflicts.astype(np.int8)
    
    # Get Fiedler vector for spectral initialization
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    degree = conflicts.sum(axis=1)
    L = np.diag(degree) - conflicts.astype(np.float64)
    L_sparse = csr_matrix(L)
    
    try:
        eigvals, eigvecs = eigsh(L_sparse, k=3, sigma=0, which='LM')
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
    except Exception:
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(N4)
    
    best_selected = None
    best_count = 0
    
    def add_free_vertices(selected, conflicts_int, N4):
        """Add all vertices that don't conflict with any selected vertex."""
        changed = True
        while changed:
            changed = False
            non_sel = np.where(~selected)[0]
            for v in non_sel:
                if np.any(conflicts_int[v] & selected):
                    continue
                selected[v] = True
                changed = True
        return selected
    
    def greedy_min_degree(conflicts, N4, seed=42):
        """Greedy minimum degree with incremental updates."""
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        rng = np.random.default_rng(seed)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (np.arange(N4) == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        
        return selected
    
    def greedy_ordering(conflicts, N4, order):
        """Greedy selection with a given vertex ordering."""
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        
        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
            alive[to_remove] = False
        
        return selected
    
    def kempe_chain_improve(selected, conflicts, N4, max_iters=1000, time_limit=None):
        """
        Improve an independent set using Kempe-chain-like swaps.
        For each non-selected vertex v with exactly one selected neighbor u,
        try swapping u out and v in, then add all newly free vertices.
        """
        selected = selected.copy()
        current_count = selected.sum()
        
        if time_limit is None:
            time_limit = deadline - 10
        
        for iteration in range(max_iters):
            if time.time() > time_limit:
                break
            
            improved = False
            best_swap = None
            best_gain = 0
            
            sel_idx = np.where(selected)[0]
            non_sel_idx = np.where(~selected)[0]
            
            if len(non_sel_idx) == 0:
                break
            
            # Find non-selected vertices with exactly 1 selected neighbor
            for v in non_sel_idx:
                sel_neighbors = np.where(conflicts[v] & selected)[0]
                if len(sel_neighbors) != 1:
                    continue
                u = sel_neighbors[0]
                
                # Tentative swap
                selected[u] = False
                selected[v] = True
                
                # Add all free vertices
                added = []
                non_sel_now = np.where(~selected)[0]
                for w in non_sel_now:
                    if np.any(conflicts[w] & selected):
                        continue
                    selected[w] = True
                    added.append(w)
                
                gain = len(added)  # net gain: -1 + 1 + len(added) = len(added)
                
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (u, v, added.copy())
                
                # Revert
                selected[u] = True
                selected[v] = False
                for w in added:
                    selected[w] = False
            
            if best_swap is not None and best_gain > 0:
                u, v, added = best_swap
                selected[u] = False
                selected[v] = True
                for w in added:
                    selected[w] = True
                current_count = selected.sum()
                improved = True
            
            # If no swap improved, try removing a high-conflict selected vertex
            if not improved:
                sel_degrees = (conflicts[sel_idx] & selected).sum(axis=1) if len(sel_idx) > 0 else np.array([])
                if len(sel_degrees) > 0:
                    max_deg_idx = np.argmax(sel_degrees)
                    u = sel_idx[max_deg_idx]
                    if sel_degrees[max_deg_idx] > 0:
                        selected[u] = False
                        added = []
                        non_sel_now = np.where(~selected)[0]
                        for w in non_sel_now:
                            if np.any(conflicts[w] & selected):
                                continue
                            selected[w] = True
                            added.append(w)
                        if len(added) >= 1:
                            current_count = selected.sum()
                            improved = True
                        else:
                            selected[u] = True
            
            if not improved:
                break
        
        return selected
    
    # Strategy 1: Greedy minimum degree
    selected = greedy_min_degree(conflicts, N4, seed=42)
    selected = add_free_vertices(selected, conflicts_int, N4)
    count = selected.sum()
    if count > best_count:
        best_count = count
        best_selected = selected.copy()
    
    # Strategy 2: Spectral ordering (most negative Fiedler first)
    order = np.argsort(fiedler)
    selected = greedy_ordering(conflicts, N4, order)
    selected = add_free_vertices(selected, conflicts_int, N4)
    count = selected.sum()
    if count > best_count:
        best_count = count
        best_selected = selected.copy()
    
    # Strategy 3: Spectral ordering (most positive Fiedler first)
    order = np.argsort(-fiedler)
    selected = greedy_ordering(conflicts, N4, order)
    selected = add_free_vertices(selected, conflicts_int, N4)
    count = selected.sum()
    if count > best_count:
        best_count = count
        best_selected = selected.copy()
    
    # Strategy 4: Kempe-chain improvement on best so far
    if best_selected is not None:
        selected = kempe_chain_improve(best_selected, conflicts, N4, max_iters=500, time_limit=deadline - 15)
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Strategy 5: Multiple random restarts with incremental degree updates
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        seed = 100 + restart
        selected = greedy_min_degree(conflicts, N4, seed=seed)
        selected = add_free_vertices(selected, conflicts_int, N4)
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Strategy 6: Kempe-chain improvement on best after restarts
    if best_selected is not None:
        selected = kempe_chain_improve(best_selected, conflicts, N4, max_iters=500, time_limit=deadline - 5)
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Strategy 7: Final polish - add all free vertices
    if best_selected is not None:
        selected = add_free_vertices(best_selected, conflicts_int, N4)
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors (they're all compatible with V4 and each other)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verify the constraint: min pairwise distance >= max norm
    if len(final_vectors) > 1:
        norms = np.sqrt(np.sum(final_vectors.astype(np.float64)**2, axis=1))
        max_norm = np.max(norms)
        
        n_final = len(final_vectors)
        valid = np.ones(n_final, dtype=bool)
        
        batch_size = 500
        for i in range(0, n_final, batch_size):
            if time.time() > deadline - 2:
                break
            batch = final_vectors[i:i+batch_size]
            if len(batch) == 0:
                break
            dots_batch = batch @ final_vectors.T
            norms_sq = np.sum(batch.astype(np.float64)**2, axis=1)
            all_norms_sq = np.sum(final_vectors.astype(np.float64)**2, axis=1)
            dist_sq = norms_sq[:, None] + all_norms_sq[None, :] - 2 * dots_batch
            
            for r in range(len(batch)):
                idx_in_final = i + r
                if not valid[idx_in_final]:
                    continue
                d_sq = dist_sq[r, :]
                d_sq[idx_in_final] = np.inf
                for c in range(n_final):
                    if c == idx_in_final:
                        continue
                    if not valid[c]:
                        continue
                    if d_sq[c] < max_norm**2 - 1e-6:
                        valid[idx_in_final] = False
                        valid[c] = False
                        break
        
        final_vectors = final_vectors[valid]
    
    return final_vectors
