"""Maximum Cut via GW Rounding, Batched Local Search, and Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses Goemans-Williamson style rounding on adjacency eigenvectors,
    batched non-conflicting local search, and simulated annealing refinement.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Symmetrize
    adj_sym = ((adj + adj.T) // 2).astype(np.int8)
    degrees = adj_sym.sum(axis=1).astype(np.int32)
    neighbors_list = [np.where(adj_sym[v] == 1)[0] for v in range(n)]

    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def compute_cut(labels):
        in0 = (labels == 0).astype(np.int32)
        in1 = (labels == 1).astype(np.int32)
        return int(in0 @ adj_sym @ in1)

    def init_same_diff(labels):
        """Initialize n_same and n_diff arrays."""
        n_same = np.zeros(n, dtype=np.int32)
        n_diff = np.zeros(n, dtype=np.int32)
        for v in range(n):
            nbrs = neighbors_list[v]
            if len(nbrs) == 0:
                continue
            same_count = int(np.sum(labels[nbrs] == labels[v]))
            n_same[v] = same_count
            n_diff[v] = len(nbrs) - same_count
        return n_same, n_diff

    def batched_local_search(labels, deadline):
        """
        Batched non-conflicting flip local search.
        In each round, find all vertices with positive gain, then apply
        non-conflicting flips (no two flipped vertices are adjacent) simultaneously.
        """
        n_same, n_diff = init_same_diff(labels)
        current_cut = int(np.sum(n_diff) // 2)
        
        max_rounds = 5000
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Compute gain for all vertices: gain[v] = 2*n_same[v] - degrees[v]
            gain = 2 * n_same - degrees
            
            # Find all vertices with positive gain
            positive = np.where(gain > 0)[0]
            if len(positive) == 0:
                break
            
            # Greedily select non-conflicting set from positive-gain vertices
            # Sort by gain descending to prioritize high-gain flips
            positive_sorted = positive[np.argsort(-gain[positive])]
            
            flipped = set()
            to_flip = []
            for v in positive_sorted:
                if v in flipped:
                    continue
                # Check if any neighbor is already in to_flip
                conflict = False
                for u in neighbors_list[v]:
                    if u in flipped:
                        conflict = True
                        break
                if not conflict:
                    to_flip.append(v)
                    flipped.add(v)
            
            if not to_flip:
                break
            
            # Apply all flips in to_flip simultaneously
            for v in to_flip:
                old_label = labels[v]
                labels[v] = 1 - old_label
                
                # Swap n_same and n_diff for v
                n_same[v], n_diff[v] = n_diff[v], n_same[v]
                
                # Update neighbors
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    # Neighbors with same label as OLD label of v now have diff
                    same_old = nbrs[labels[nbrs] == old_label]
                    if len(same_old) > 0:
                        n_same[same_old] -= 1
                        n_diff[same_old] += 1
                    
                    # Neighbors with different label from OLD label of v now have same
                    diff_old = nbrs[labels[nbrs] != old_label]
                    if len(diff_old) > 0:
                        n_same[diff_old] += 1
                        n_diff[diff_old] -= 1
                
                current_cut += gain[v] // 2
        
        return labels, current_cut

    def simulated_annealing(labels, deadline, n_iter_cap=200000):
        """Simulated annealing refinement to escape local optima."""
        n_same, n_diff = init_same_diff(labels)
        current_cut = int(np.sum(n_diff) // 2)
        best_cut = current_cut
        best_labels = labels.copy()
        
        T = 5.0
        cooling_rate = 0.99995
        min_T = 0.05
        
        rng = np.random.default_rng(12345)
        iter_count = 0
        
        while time.time() < deadline and T > min_T and iter_count < n_iter_cap:
            v = rng.integers(0, n)
            
            if degrees[v] == 0:
                T *= cooling_rate
                iter_count += 1
                continue
            
            # Delta if we flip v: new_n_diff - old_n_diff = n_same[v] - n_diff[v]
            delta = n_same[v] - n_diff[v]
            
            accept = False
            if delta >= 0:
                accept = True
            elif T > 0:
                if rng.random() < np.exp(delta / T):
                    accept = True
            
            if accept:
                old_label = labels[v]
                labels[v] = 1 - old_label
                
                # Swap for v
                n_same[v], n_diff[v] = n_diff[v], n_same[v]
                
                # Update neighbors
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    same_old = nbrs[labels[nbrs] == old_label]
                    if len(same_old) > 0:
                        n_same[same_old] -= 1
                        n_diff[same_old] += 1
                    
                    diff_old = nbrs[labels[nbrs] != old_label]
                    if len(diff_old) > 0:
                        n_same[diff_old] += 1
                        n_diff[diff_old] -= 1
                
                current_cut += delta // 2
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T *= cooling_rate
            iter_count += 1
        
        return best_labels, best_cut

    def pair_flip_local_search(labels, deadline):
        """2-flip local search: try flipping pairs of adjacent vertices."""
        n_same, n_diff = init_same_diff(labels)
        current_cut = int(np.sum(n_diff) // 2)
        best_cut = current_cut
        best_labels = labels.copy()
        
        # Build edge list
        edges = []
        for v in range(n):
            for u in neighbors_list[v]:
                if u > v:
                    edges.append((v, u))
        
        max_rounds = 200
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            # Shuffle edges for diversity
            rng_local = np.random.default_rng(rnd * 7 + 1)
            edge_indices = rng_local.permutation(len(edges))
            
            for idx in edge_indices:
                v, u = edges[idx]
                
                # Compute gain of flipping both v and u
                # If v and u are adjacent and we flip both:
                # For v: neighbors that are u change from diff to same (if they were diff) 
                # Actually, let's compute delta carefully.
                
                # Current state: labels[v], labels[u]
                # After flip: 1-labels[v], 1-labels[u]
                
                # For vertex v, its contribution to cut changes based on neighbors.
                # Flipping v alone: delta_v = n_same[v] - n_diff[v]
                # But flipping u also affects v's n_same/n_diff if u is a neighbor.
                
                # Let's compute the combined effect:
                # Let lv = labels[v], lu = labels[u]
                # After flip: lv' = 1-lv, lu' = 1-lu
                
                # For v's neighbors (excluding u): 
                #   those with label == lv become diff, those with label != lv become same
                # For v's neighbor u: 
                #   if lv == lu, after flip lv' == lu' still, so same stays same
                #   if lv != lu, after flip lv' != lu' still, so diff stays diff
                # So u's flip doesn't change v's relationship with u!
                
                # Wait, that's not right. Let me reconsider.
                # v's n_same counts neighbors with same label as v.
                # If we flip v, v's label changes. u's label also changes.
                # The relationship between v and u: 
                #   Before: same if lv==lu, diff if lv!=lu
                #   After: same if (1-lv)==(1-lu) i.e. lv==lu, diff if lv!=lu
                # So the v-u edge status is unchanged!
                
                # Therefore, flipping both v and u:
                # For v: all neighbors except u flip their relative status with v
                #   n_same_excl_u = number of neighbors w != u with labels[w] == lv
                #   n_diff_excl_u = (degrees[v]-1) - n_same_excl_u  (excluding u)
                #   After flip v: n_same_new_v = n_diff_excl_u + (1 if lu==lv else 0) 
                #   Wait, u's label also flips, so u's relationship with v is preserved.
                #   So n_same_new_v = (neighbors of v with label != lv, excluding u) + (1 if lu==lv else 0)
                #   = n_diff_excl_u + (1 if lu==lv else 0)
                #   And n_diff_new_v = n_same_excl_u + (1 if lu!=lv else 0)
                
                # This is getting complex. Let me just compute delta directly.
                
                lv = labels[v]
                lu = labels[u]
                
                # n_same[v] includes u if lu==lv
                same_v_incl_u = int(np.sum(labels[neighbors_list[v]] == lv))
                # This is n_same[v]
                
                # For the pair flip, the gain is:
                # delta_total = (new_n_diff[v] + new_n_diff[u]) - (old_n_diff[v] + old_n_diff[u])
                # but we need to be careful about double counting.
                
                # Actually, total cut = sum(n_diff)/2
                # delta_cut = (delta_n_diff[v] + delta_n_diff[u]) / 2
                # where delta_n_diff[v] = new_n_diff[v] - old_n_diff[v]
                
                # For v: old_n_diff[v] = n_diff[v]
                # After flipping v and u:
                #   v's new label = 1-lv
                #   For neighbor w of v:
                #     if w == u: new label of u = 1-lu. Same as new v if (1-lv)==(1-lu) i.e. lv==lu
                #     if w != u: new label of w unchanged. Same as new v if labels[w]==(1-lv) i.e. labels[w]!=lv
                
                # new_n_diff[v] = count of w in neighbors(v) where labels[w] != (1-labels[v])
                #                = count of w where labels[w] == labels[v]  [for w != u]
                #                + (1 if labels[u] == labels[v] else 0)  [for w == u]
                #                = (n_same[v] - (1 if lu==lv else 0)) + (1 if lu==lv else 0)
                #                = n_same[v]
                
                # Wait, that means new_n_diff[v] = n_same[v]? That's the same as single flip!
                # Let me verify: if I flip only v, new_n_diff[v] = n_same[v]. Yes.
                # If I flip both v and u, for v's perspective, u's label also flipped,
                # so the v-u edge status is preserved. All other neighbors have same status
                # as if only v flipped. So new_n_diff[v] when flipping both = n_same[v].
                
                # Similarly new_n_diff[u] = n_same[u].
                
                # So delta_cut = (n_same[v] - n_diff[v] + n_same[u] - n_diff[u]) / 2
                #              = (gain[v] + gain[u]) / 2  where gain[x] = n_same[x] - n_diff[x]
                
                delta_cut = (n_same[v] - n_diff[v] + n_same[u] - n_diff[u]) / 2.0
                
                if delta_cut > 0:
                    # Apply the pair flip
                    labels[v] = 1 - lv
                    labels[u] = 1 - lu
                    
                    # Update n_same, n_diff for v
                    n_same[v], n_diff[v] = n_diff[v], n_same[v]
                    # Update n_same, n_diff for u
                    n_same[u], n_diff[u] = n_diff[u], n_same[u]
                    
                    # Update neighbors of v (excluding u)
                    for w in neighbors_list[v]:
                        if w == u:
                            continue
                        if labels[w] == lv:  # same as old v, now diff
                            n_same[w] -= 1
                            n_diff[w] += 1
                        else:  # diff from old v, now same
                            n_same[w] += 1
                            n_diff[w] -= 1
                    
                    # Update neighbors of u (excluding v)
                    for w in neighbors_list[u]:
                        if w == v:
                            continue
                        if labels[w] == lu:  # same as old u, now diff
                            n_same[w] -= 1
                            n_diff[w] += 1
                        else:  # diff from old u, now same
                            n_same[w] += 1
                            n_diff[w] -= 1
                    
                    current_cut = int(current_cut + delta_cut)
                    improved = True
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
            
            if not improved:
                break
        
        return best_labels, best_cut

    # Generate diverse initial solutions via GW-style eigenvector rounding
    best_labels = None
    best_cut = -1

    try:
        adj_float = adj_sym.astype(np.float64)
        eigvals, eigvecs = np.linalg.eigh(adj_float)
        
        # Try multiple k values for diversity
        k_values = [5, 10, 20, 30]
        k_values = [k for k in k_values if k <= n]
        if not k_values:
            k_values = [n]
        
        rng = np.random.default_rng(42)
        candidates = []
        
        for k in k_values:
            V = eigvecs[:, -k:]  # top k eigenvectors
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            V_norm = V / norms
            
            num_rounds = 500
            for i in range(num_rounds):
                if time.time() > deadline - 60:
                    break
                
                u = rng.standard_normal(k)
                u /= np.linalg.norm(u)
                proj = V_norm @ u
                labels = (proj > 0).astype(np.int8)
                
                if np.all(labels == 0) or np.all(labels == 1):
                    continue
                
                cut = compute_cut(labels)
                candidates.append((cut, labels.copy()))
        
        # Add random and greedy inits
        random_labels = rng.integers(0, 2, size=n, dtype=np.int8)
        candidates.append((compute_cut(random_labels), random_labels))
        
        greedy_labels = np.zeros(n, dtype=np.int8)
        order = np.argsort(-degrees)
        for v in order:
            nbrs = neighbors_list[v]
            if len(nbrs) == 0:
                continue
            count_1 = np.sum(greedy_labels[nbrs] == 1)
            count_0 = np.sum(greedy_labels[nbrs] == 0)
            greedy_labels[v] = 1 if count_1 > count_0 else 0
        candidates.append((compute_cut(greedy_labels), greedy_labels))
        
        # Sort by cut size, keep top diverse candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Deduplicate by keeping only distinct label patterns
        seen = set()
        unique_candidates = []
        for cut, lbls in candidates:
            key = lbls.tobytes()
            if key not in seen:
                seen.add(key)
                unique_candidates.append((cut, lbls))
        
        # Process top candidates with batched LS + SA
        max_candidates = 30
        for cut_est, labels in unique_candidates[:max_candidates]:
            if time.time() > deadline - 10:
                break
            
            labels = labels.copy()
            
            # Phase 1: Batched local search
            ls_labels, ls_cut = batched_local_search(labels, deadline - 20)
            
            # Phase 2: Pair-flip local search
            pf_labels, pf_cut = pair_flip_local_search(ls_labels, deadline - 10)
            
            # Phase 3: Simulated annealing
            sa_labels, sa_cut = simulated_annealing(pf_labels, deadline - 5, n_iter_cap=100000)
            
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
                
            # Also track the best from each phase
            if pf_cut > best_cut:
                best_cut = pf_cut
                best_labels = pf_labels.copy()
            if ls_cut > best_cut:
                best_cut = ls_cut
                best_labels = ls_labels.copy()
                
    except Exception:
        # Fallback
        rng = np.random.default_rng(42)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            labels = rng.integers(0, 2, size=n, dtype=np.int8)
            ls_labels, ls_cut = batched_local_search(labels, deadline - 5)
            if ls_cut > best_cut:
                best_cut = ls_cut
                best_labels = ls_labels.copy()

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
