import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

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

    # Precompute neighbor lists
    neighbor_lists = []
    for i in range(N4):
        neighbor_lists.append(np.where(conflicts[i])[0])

    def greedy_mis():
        alive = np.ones(N4, dtype=bool)
        selected = []
        degrees = conflicts.sum(axis=1).astype(np.int64)
        rng = np.random.default_rng(random.randint(0, 2**31 - 1))

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]

            selected.append(chosen)

            neighbors = neighbor_lists[chosen]
            to_remove = np.concatenate(([chosen], neighbors[alive[neighbors]]))
            alive[to_remove] = False

            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                removed_neighbor_counts = conflicts[remaining][:, to_remove].sum(axis=1)
                degrees[remaining] -= removed_neighbor_counts

        return selected

    def spectral_partition():
        """Use Fiedler vector to partition vertices into two independent sets."""
        from scipy.sparse.csgraph import laplacian as sparse_laplacian
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        
        # Build sparse adjacency
        adj = np.zeros((N4, N4), dtype=np.int8)
        for i in range(N4):
            adj[i, neighbor_lists[i]] = 1
        adj = csr_matrix(adj)
        
        # Compute Laplacian
        L = sparse_laplacian(adj)
        
        # Get Fiedler vector (second smallest eigenvalue)
        try:
            eigvals, eigvecs = eigsh(L, k=2, which='SM')
            fiedler = eigvecs[:, 1]
        except:
            # Fallback: random partition
            rng = np.random.default_rng(42)
            fiedler = rng.standard_normal(N4)
        
        # Partition by sign of Fiedler vector
        partition_a = np.where(fiedler >= 0)[0]
        partition_b = np.where(fiedler < 0)[0]
        
        # For each partition, greedily build independent set
        def greedy_from_partition(partition):
            if len(partition) == 0:
                return []
            alive = np.zeros(N4, dtype=bool)
            alive[partition] = True
            selected = []
            degrees = conflicts.sum(axis=1).astype(np.int64)
            rng = np.random.default_rng(random.randint(0, 2**31 - 1))
            
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                
                alive_degrees = degrees[alive_indices]
                min_deg = np.min(alive_degrees)
                cands = alive_indices[alive_degrees == min_deg]
                chosen = cands[rng.integers(len(cands))]
                
                selected.append(chosen)
                
                neighbors = neighbor_lists[chosen]
                to_remove = np.concatenate(([chosen], neighbors[alive[neighbors]]))
                alive[to_remove] = False
                
                remaining = np.where(alive)[0]
                if len(remaining) > 0:
                    removed_neighbor_counts = conflicts[remaining][:, to_remove].sum(axis=1)
                    degrees[remaining] -= removed_neighbor_counts
            
            return selected
        
        set_a = greedy_from_partition(partition_a)
        set_b = greedy_from_partition(partition_b)
        
        # Try to merge: add vertices from one set that don't conflict with the other
        def try_merge(set1, set2):
            selected = list(set1)
            selected_set = set(selected)
            conflict_mask = np.zeros(N4, dtype=bool)
            for s in selected:
                conflict_mask[neighbor_lists[s]] = True
            conflict_mask[list(selected)] = True
            
            for v in set2:
                if not conflict_mask[v]:
                    selected.append(v)
                    conflict_mask[v] = True
                    conflict_mask[neighbor_lists[v]] = True
            
            return selected
        
        merged_ab = try_merge(set_a, set_b)
        merged_ba = try_merge(set_b, set_a)
        
        best = max([set_a, set_b, merged_ab, merged_ba], key=len)
        return best

    def kempe_local_search(selected):
        """Kempe-chain style local search: try to improve by swapping small sets."""
        selected = list(selected)
        selected_set = set(selected)
        
        # Build conflict mask
        conflict_mask = np.zeros(N4, dtype=bool)
        for s in selected:
            conflict_mask[neighbor_lists[s]] = True
        conflict_mask[list(selected)] = True
        
        improved = True
        max_iterations = 50
        for it in range(max_iterations):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            improved = False
            
            # Find addable vertices
            addable = np.where(~conflict_mask)[0]
            
            # Try single additions
            for a in addable:
                if not conflict_mask[a]:
                    selected.append(a)
                    conflict_mask[a] = True
                    conflict_mask[neighbor_lists[a]] = True
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one selected, try to add one or more
            for s in selected:
                if time.time() > deadline - 5:
                    break
                
                # Temporarily remove s
                new_conflict_mask = np.zeros(N4, dtype=bool)
                for t in selected:
                    if t != s:
                        new_conflict_mask[neighbor_lists[t]] = True
                new_conflict_mask[list(selected)] = True
                new_conflict_mask[s] = True  # s freed up
                
                # Find what can be added now
                can_add = np.where(~new_conflict_mask)[0]
                can_add = can_add[can_add != s]
                
                if len(can_add) > 0:
                    # Try adding multiple
                    new_selected = [t for t in selected if t != s]
                    new_mask = new_conflict_mask.copy()
                    added_count = 0
                    for a in can_add:
                        if not new_mask[a]:
                            new_selected.append(a)
                            new_mask[a] = True
                            new_mask[neighbor_lists[a]] = True
                            added_count += 1
                    
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        conflict_mask = new_mask
                        improved = True
                        break
        
        return selected

    # Strategy 1: Run multiple greedy passes
    best_selected = []
    num_greedy = 30
    for _ in range(num_greedy):
        if time.time() > deadline - 20:
            break
        sel = greedy_mis()
        if len(sel) > len(best_selected):
            best_selected = sel

    # Strategy 2: Spectral partitioning
    if time.time() < deadline - 15:
        spectral_result = spectral_partition()
        if len(spectral_result) > len(best_selected):
            best_selected = spectral_result

    # Strategy 3: Kempe local search on best result
    if time.time() < deadline - 10:
        best_selected = kempe_local_search(best_selected)

    # Final: try to add any remaining non-conflicting vertices
    selected_set = set(best_selected)
    conflicts_with_selected = np.zeros(N4, dtype=bool)
    for s in best_selected:
        conflicts_with_selected[neighbor_lists[s]] = True
    conflicts_with_selected[list(best_selected)] = True

    unselected = np.where(~conflicts_with_selected)[0]
    for u in unselected:
        if time.time() > deadline - 5:
            break
        if not conflicts_with_selected[u]:
            best_selected.append(u)
            conflicts_with_selected[u] = True
            conflicts_with_selected[neighbor_lists[u]] = True

    result_v4 = vectors_v4[best_selected]

    # All 22 V1 vectors are mutually compatible and compatible with all V4 vectors
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
