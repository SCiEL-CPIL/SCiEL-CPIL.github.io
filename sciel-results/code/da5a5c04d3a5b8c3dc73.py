import numpy as np
import itertools
import time

D = 11
N4_TOTAL = 5302

def construct():
    start_time = time.time()
    deadline = start_time + 130

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

    # Convert to sparse for efficiency
    from scipy.sparse import csr_matrix
    conflicts_sparse = csr_matrix(conflicts)

    best_selected = None
    best_count = 0

    # Strategy 1: Greedy minimum degree with multiple random seeds
    for seed in [42, 123, 456, 789, 101]:
        if time.time() > deadline - 30:
            break

        rng = np.random.default_rng(seed)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        # Compute current degrees
        degrees = conflicts.sum(axis=1).astype(np.int64).copy()

        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]

            selected[chosen] = True
            # Remove chosen and its neighbors from alive
            to_remove = np.where(conflicts[chosen] | (np.arange(N4) == chosen))[0]
            alive[to_remove] = False
            # Update degrees for remaining alive vertices
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                # For each remaining vertex, count how many of its neighbors are in to_remove
                # This is the number of neighbors that just died
                deg_updates = conflicts[remaining][:, to_remove].sum(axis=1)
                degrees[remaining] -= deg_updates

        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # Strategy 2: Spectral initialization + greedy
    from scipy.sparse.linalg import eigsh
    L = np.diag(conflicts.sum(axis=1)) - conflicts.astype(np.float64)
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

    # Order by fiedler value
    order = np.argsort(fiedler)
    alive = np.ones(N4, dtype=bool)
    selected = np.zeros(N4, dtype=bool)

    for i in order:
        if not alive[i]:
            continue
        selected[i] = True
        to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
        alive[to_remove] = False

    count = selected.sum()
    if count > best_count:
        best_count = count
        best_selected = selected.copy()

    # Strategy 3: Kempe-chain refinement on the best solution
    if best_selected is not None:
        selected = best_selected.copy()
        # Try to improve via Kempe-chain style swaps
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            # Find vertices not in selected that have no conflicts with selected
            non_selected = np.where(~selected)[0]
            if len(non_selected) == 0:
                break
            
            # Find vertices in selected that could be removed
            # We want to swap: remove one from selected, add one not in selected
            # But only if the one not in selected has fewer conflicts with remaining selected
            
            # Simpler: just try to add any vertex that doesn't conflict
            added_any = False
            for v in non_selected:
                if not np.any(conflicts[v] & selected):
                    selected[v] = True
                    added_any = True
            
            if not added_any:
                break
        
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # Strategy 4: Spectral partitioning with multiple eigenvectors + local search
    if time.time() < deadline - 20:
        try:
            # Get more eigenvectors for better partitioning
            eigvals, eigvecs = eigsh(L_sparse, k=5, sigma=0, which='LM')
            idx = np.argsort(eigvals)
            eigvecs = eigvecs[:, idx]
            
            # Use the second eigenvector (Fiedler) for initial partition
            fiedler = eigvecs[:, 1]
            
            # Try multiple threshold values for partitioning
            for threshold in [np.median(fiedler), np.percentile(fiedler, 30), np.percentile(fiedler, 70)]:
                if time.time() > deadline - 15:
                    break
                
                # Partition: left = fiedler < threshold, right = fiedler >= threshold
                left_mask = fiedler < threshold
                right_mask = ~left_mask
                
                # Greedily select from each partition
                for part_mask in [left_mask, right_mask]:
                    alive = part_mask.copy()
                    sel = np.zeros(N4, dtype=bool)
                    
                    # Order by fiedler value within the partition
                    part_indices = np.where(part_mask)[0]
                    part_order = part_indices[np.argsort(fiedler[part_indices])]
                    
                    for i in part_order:
                        if not alive[i]:
                            continue
                        sel[i] = True
                        to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
                        alive[to_remove] = False
                    
                    count = sel.sum()
                    if count > best_count:
                        best_count = count
                        best_selected = sel.copy()
                        
                        # Local search refinement
                        if time.time() < deadline - 10:
                            non_selected = np.where(~sel)[0]
                            for v in non_selected:
                                if not np.any(conflicts[v] & sel):
                                    sel[v] = True
                            
                            count = sel.sum()
                            if count > best_count:
                                best_count = count
                                best_selected = sel.copy()
        except Exception:
            pass

    # Strategy 5: More random restarts with spectral ordering
    for restart in range(15):
        if time.time() > deadline - 10:
            break

        rng2 = np.random.default_rng(1000 + restart * 7)
        
        # Mix of random and spectral ordering
        if restart % 2 == 0:
            order = rng2.permutation(N4)
        else:
            # Perturb the fiedler ordering
            noise = rng2.standard_normal(N4) * 0.1
            perturbed_fiedler = fiedler + noise
            order = np.argsort(perturbed_fiedler)

        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)

        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
            alive[to_remove] = False

        count = selected.sum()
        
        # Local search refinement
        if time.time() < deadline - 5:
            non_selected = np.where(~selected)[0]
            for v in non_selected:
                if not np.any(conflicts[v] & selected):
                    selected[v] = True
            
            count = selected.sum()

        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    result_v4 = vectors_v4[best_selected]

    # Add all V1 vectors
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
