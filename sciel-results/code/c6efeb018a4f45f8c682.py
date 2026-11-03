"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate Candidate Pools
    
    # Type A: +/- 2e_i (22 vectors)
    type_a_list = []
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign * 2
            type_a_list.append(v)
    type_a = np.array(type_a_list, dtype=np.int8)
    
    # Type B: +/- e_i +/- e_j +/- e_k +/- e_l (5280 vectors)
    type_b_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_b_list.append(v)
    type_b = np.array(type_b_list, dtype=np.int8)
    n_b = len(type_b)
    
    # 2. Precompute Conflict Matrix for Type B
    c_b = type_b.astype(np.int32)
    dot_b = c_b @ c_b.T
    conflict_b = (dot_b > 2).astype(np.uint8)
    np.fill_diagonal(conflict_b, 0)
    
    best_count_b = 0
    best_selected_b = np.zeros(n_b, dtype=np.uint8)
    
    # 3. Spectral Embedding for Cluster Identification
    # Build compatibility graph adjacency
    compat_b = (1 - conflict_b).astype(np.float64)
    np.fill_diagonal(compat_b, 0)
    
    # Compute graph Laplacian L = D - A
    degree_vec = compat_b.sum(axis=1)
    L = np.diag(degree_vec) - compat_b
    
    # Use scipy for eigenvalue decomposition if available, else use numpy
    try:
        from scipy.sparse.linalg import eigs
        import scipy.sparse as sp
        L_sparse = sp.csr_matrix(L)
        # Find smallest eigenvalues (most connected clusters)
        k_eig = min(20, n_b - 1)
        eigenvalues, eigenvectors = eigs(L_sparse, k=k_eig, sigma=0, which='LM')
        # Take real part
        eigenvectors = np.real(eigenvectors)
        # Normalize columns
        norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        norms[norms == 0] = 1
        eigenvectors = eigenvectors / norms
    except Exception:
        # Fallback: use degree-based ranking
        eigenvectors = None
    
    # 4. Multi-Start Search on Type B
    rng = np.random.default_rng(42)
    
    def get_cluster_priority():
        """Get priority order based on spectral embedding or degree."""
        if eigenvectors is not None:
            # Cluster nodes by their position in spectral space
            # Use first few eigenvectors to assign cluster labels
            k_clusters = min(10, n_b // 100 + 1)
            from scipy.cluster.vq import kmeans2
            try:
                # Use first 3 eigenvectors for clustering
                embed = eigenvectors[:, :min(3, eigenvectors.shape[1])]
                # Normalize
                norms = np.linalg.norm(embed, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embed = embed / norms
                # Simple k-means
                centers, labels = kmeans2(embed, k_clusters, minit='points')
                # Priority: nodes in larger clusters first, then by degree within cluster
                cluster_sizes = np.bincount(labels, minlength=k_clusters)
                # Sort clusters by size descending
                cluster_order = np.argsort(-cluster_sizes)
                # Assign priority
                priority = np.zeros(n_b)
                for rank, cl in enumerate(cluster_order):
                    mask = labels == cl
                    deg_in_cluster = degree_vec[mask]
                    # Within cluster, prioritize high degree
                    local_priority = np.argsort(-deg_in_cluster)
                    for i, idx in enumerate(np.where(mask)[0]):
                        priority[idx] = (k_clusters - rank) * 1000 - i
                return priority
            except Exception:
                pass
        # Fallback: use degree
        return degree_vec.copy()
    
    cluster_priority = get_cluster_priority()
    
    def greedy_construct_b(seed_nodes=None):
        """
        Greedy selection using minimum-degree heuristic, optionally seeded.
        """
        selected = np.zeros(n_b, dtype=np.uint8)
        remaining = np.ones(n_b, dtype=np.uint8)
        
        # If seed nodes provided, start with them
        if seed_nodes is not None and len(seed_nodes) > 0:
            for sn in seed_nodes:
                if remaining[sn]:
                    selected[sn] = 1
                    remaining[sn] = 0
                    remaining[conflict_b[sn] == 1] = 0
        
        while True:
            if time.time() > deadline - 10.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree
            deg = (conflict_b * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n_b + 1
            
            min_deg = deg.min()
            if min_deg >= n_b:
                break
                
            candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                break
            
            # Use cluster priority to break ties
            if len(candidates_to_pick) > 1:
                cand_priorities = cluster_priority[candidates_to_pick]
                # Pick from top priority candidates
                top_k = min(5, len(candidates_to_pick))
                top_indices = np.argsort(-cand_priorities)[:top_k]
                pick_idx = candidates_to_pick[top_indices[rng.integers(top_k)]]
            else:
                pick_idx = candidates_to_pick[0]
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            remaining[conflict_b[pick_idx] == 1] = 0
            
        return selected

    def spectral_cluster_search():
        """
        Use spectral embedding to identify dense regions and greedily select.
        """
        if time.time() > deadline - 30.0:
            return np.zeros(n_b, dtype=np.uint8)
        
        best_selected = np.zeros(n_b, dtype=np.uint8)
        best_count = 0
        
        # Try seeds from top priority nodes
        top_seeds = np.argsort(-cluster_priority)[:30]
        
        for start_node in top_seeds:
            if time.time() > deadline - 10.0:
                break
            
            # Greedy expansion from this seed node
            selected = greedy_construct_b(seed_nodes=[start_node])
            
            count = selected.sum()
            if count > best_count:
                best_count = count
                best_selected = selected.copy()
        
        return best_selected

    def local_search_b(selected):
        """
        Refine the selected set using 1-for-2 moves and 2-for-3 moves.
        """
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        improved = True
        max_moves = 200
        move_count = 0
        moves_made = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Compute conflict with S
            conflict_with_S = conflict_b[:, sel_indices]
            total_inc = conflict_with_S.sum(axis=1)
            
            # base_mask: nodes compatible with ALL S and not selected
            base_mask = (total_inc == 0) & (selected == 0)
            
            # Try 1-for-2 moves
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v = sel_indices[idx]
                
                # Cond: conflicts ONLY with v
                cond_mask = (total_inc == 1) & (conflict_with_S[:, idx] == 1) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                sub_conflict = conflict_b[np.ix_(cand_indices, cand_indices)]
                np.fill_diagonal(sub_conflict, 1)
                
                # Find a pair of non-conflicting candidates
                zero_rows, zero_cols = np.where(sub_conflict == 0)
                if len(zero_rows) > 0:
                    u = cand_indices[zero_rows[0]]
                    w = cand_indices[zero_cols[0]]
                    
                    selected[v] = 0
                    selected[u] = 1
                    selected[w] = 1
                    improved = True
                    moves_made += 1
                    break
                        
        # Try 2-for-3 moves
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) >= 2:
            improved2 = True
            max_moves2 = 50
            move_count2 = 0
            
            while improved2 and move_count2 < max_moves2 and time.time() < deadline - 3.0:
                improved2 = False
                move_count2 += 1
                
                sel_indices = np.where(selected == 1)[0]
                k_sel = len(sel_indices)
                if k_sel < 2:
                    break
                
                # Pick two random selected nodes to remove
                pair_idx = rng.choice(k_sel, size=2, replace=False)
                v1 = sel_indices[pair_idx[0]]
                v2 = sel_indices[pair_idx[1]]
                
                # Nodes compatible with S \ {v1, v2}
                conflict_with_rest = conflict_b[:, np.setdiff1d(sel_indices, [v1, v2])]
                if conflict_with_rest.shape[1] > 0:
                    total_inc_rest = conflict_with_rest.sum(axis=1)
                else:
                    total_inc_rest = np.zeros(n_b, dtype=np.int32)
                
                base_mask2 = (total_inc_rest == 0) & (selected == 0)
                # Also need to be compatible with v1 OR v2 (at least not conflict with both)
                # Actually, they just need to be compatible with S\{v1,v2}
                
                cand_indices2 = np.where(base_mask2 > 0)[0]
                
                if len(cand_indices2) < 3:
                    continue
                
                sub_conflict2 = conflict_b[np.ix_(cand_indices2, cand_indices2)]
                np.fill_diagonal(sub_conflict2, 1)
                
                # Find a triple of mutually non-conflicting candidates
                found_triple = False
                for i in range(len(cand_indices2)):
                    if found_triple:
                        break
                    for j in range(i+1, len(cand_indices2)):
                        if sub_conflict2[i, j] == 0:
                            for k in range(j+1, len(cand_indices2)):
                                if sub_conflict2[i, k] == 0 and sub_conflict2[j, k] == 0:
                                    u = cand_indices2[i]
                                    w = cand_indices2[j]
                                    x = cand_indices2[k]
                                    selected[v1] = 0
                                    selected[v2] = 0
                                    selected[u] = 1
                                    selected[w] = 1
                                    selected[x] = 1
                                    improved2 = True
                                    moves_made += 1
                                    found_triple = True
                                    break
                
                if not improved2 and move_count2 > 20:
                    break
                        
        return selected, moves_made

    # Run spectral cluster search first
    spectral_selected = spectral_cluster_search()
    spectral_count = spectral_selected.sum()
    
    if spectral_count > best_count_b:
        best_count_b = spectral_count
        best_selected_b = spectral_selected.copy()
    
    # Run multiple restarts
    max_restarts = 60
    for i in range(max_restarts):
        if time.time() > deadline - 25.0:
            break
            
        if i % 4 == 0 and best_count_b > 0:
            # Start from best solution and do local search
            selected = best_selected_b.copy()
        else:
            # Randomized greedy with cluster-guided selection
            selected = greedy_construct_b()
        
        current_count = selected.sum()
        
        selected, moves = local_search_b(selected)
        current_count = selected.sum()
        
        if current_count > best_count_b:
            best_count_b = current_count
            best_selected_b = selected.copy()
        
        # Occasionally try to improve from spectral solution
        if i % 8 == 4 and spectral_count > best_count_b - 10:
            selected = spectral_selected.copy()
            selected, moves = local_search_b(selected)
            current_count = selected.sum()
            if current_count > best_count_b:
                best_count_b = current_count
                best_selected_b = selected.copy()

    # 5. Final Polish: Try to add more vectors to the best solution
    if best_count_b > 0 and time.time() < deadline - 5.0:
        sel_indices = np.where(best_selected_b == 1)[0]
        conflict_with_S = conflict_b[:, sel_indices]
        total_inc = conflict_with_S.sum(axis=1)
        base_mask = (total_inc == 0) & (best_selected_b == 0)
        
        # Try adding vectors one by one in priority order
        cand_indices = np.where(base_mask > 0)[0]
        if len(cand_indices) > 0:
            cand_priorities = cluster_priority[cand_indices]
            order = np.argsort(-cand_priorities)
            
            for ci in order:
                if time.time() > deadline - 2.0:
                    break
                idx = cand_indices[ci]
                # Check if compatible with all selected
                if conflict_b[idx, sel_indices].sum() == 0:
                    best_selected_b[idx] = 1
                    sel_indices = np.where(best_selected_b == 1)[0]
                    best_count_b += 1

    # 4. Combine Results
    if best_count_b == 0:
        return type_a
    
    selected_type_b = type_b[best_selected_b == 1]
    
    # Concatenate
    result = np.vstack([type_a, selected_type_b])
    
    return result.astype(np.int64)

# EVOLVE-BLOCK-END
