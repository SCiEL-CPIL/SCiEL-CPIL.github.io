"""Kissing number in dimension 11: spectral relaxation + local search for large independent sets."""
import numpy as np
import itertools
import time
import random
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    D = 11

    # Generate candidate vectors
    candidates_list = []

    # Weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-3 vectors (norm^2=3): C(11,3)*8 = 1320
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-1 vectors (norm^2=1): 22
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected (i != j)
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Precompute full degree
    full_deg = conflict.sum(axis=1)

    # Spectral approach: compute Laplacian of conflict graph
    # L = D - A where A is adjacency (conflict), D is degree matrix
    # Large independent set => vertices with low values in Fiedler vector (2nd eigenvector)
    # or use max-cut spectral relaxation

    # Build sparse conflict matrix
    conflict_sparse = csr_matrix(conflict.astype(np.float64))
    degree_vec = full_deg.astype(np.float64)
    L = csr_matrix((degree_vec[:, np.newaxis] - conflict_sparse))

    # Get leading eigenvectors of Laplacian
    # For max-cut: we want to partition into 2 sets maximizing cuts
    # For independent set: use Fiedler vector to find large low-degree subset

    try:
        # Compute smallest non-trivial eigenvectors
        # k=10 to get multiple spectral directions
        k_eig = min(10, n - 1)
        eigenvalues, eigenvectors = eigsh(L, k=k_eig, which='SM', return_eigenvectors=True,
                                          v0=np.ones(n))
    except Exception:
        eigenvalues, eigenvectors = [], np.zeros((n, 1))

    best_selected = []

    # Strategy 1: Spectral-based independent set extraction
    if len(eigenvectors) > 0:
        # For each eigenvector, select vertices below median value
        # (low values in Fiedler vector tend to form independent sets)
        for ev_idx in range(eigenvectors.shape[1]):
            if time.time() > deadline - 30.0:
                break
            ev = eigenvectors[:, ev_idx]
            # Select vertices with values below the 25th percentile
            threshold_val = np.percentile(ev, 25)
            spectral_set = np.where(ev <= threshold_val)[0]

            # Refine: greedily remove conflicts from spectral set
            selected = []
            remaining = set(spectral_set.tolist())
            while remaining and time.time() < deadline - 25.0:
                # Pick vertex with fewest conflicts in remaining
                best_v = None
                best_deg = np.inf
                for v in remaining:
                    deg_in_remaining = sum(1 for u in remaining if conflict[v, u])
                    if deg_in_remaining < best_deg:
                        best_deg = deg_in_remaining
                        best_v = v
                    if deg_in_remaining == 0:
                        break
                if best_v is None:
                    break
                selected.append(best_v)
                remaining.discard(best_v)
                # Remove all conflicts of best_v
                for u in list(remaining):
                    if conflict[best_v, u]:
                        remaining.discard(u)

            if len(selected) > len(best_selected):
                best_selected = selected

        # Also try above-median selection
        for ev_idx in range(eigenvectors.shape[1]):
            if time.time() > deadline - 25.0:
                break
            ev = eigenvectors[:, ev_idx]
            threshold_val = np.percentile(ev, 75)
            spectral_set = np.where(ev >= threshold_val)[0]

            selected = []
            remaining = set(spectral_set.tolist())
            while remaining and time.time() < deadline - 20.0:
                best_v = None
                best_deg = np.inf
                for v in remaining:
                    deg_in_remaining = sum(1 for u in remaining if conflict[v, u])
                    if deg_in_remaining < best_deg:
                        best_deg = deg_in_remaining
                        best_v = v
                    if deg_in_remaining == 0:
                        break
                if best_v is None:
                    break
                selected.append(best_v)
                remaining.discard(best_v)
                for u in list(remaining):
                    if conflict[best_v, u]:
                        remaining.discard(u)

            if len(selected) > len(best_selected):
                best_selected = selected

    # Strategy 2: Min-degree greedy (as fallback / complement)
    def greedy_min_degree(remaining_init, time_limit):
        remaining = remaining_init.copy()
        selected = []
        while time.time() < time_limit and remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        return selected

    if time.time() < deadline - 20.0:
        sel = greedy_min_degree(np.ones(n, dtype=np.uint8), deadline - 20.0)
        if len(sel) > len(best_selected):
            best_selected = sel

    # Strategy 3: Randomized greedy restarts
    np.random.seed(42)
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 15.0:
            break
        frac = np.random.uniform(0.4, 1.0)
        subset_size = int(n * frac)
        subset_indices = np.random.choice(n, size=subset_size, replace=False)
        remaining_init = np.zeros(n, dtype=np.uint8)
        remaining_init[subset_indices] = 1
        sel = greedy_min_degree(remaining_init, deadline - 15.0)
        if len(sel) > len(best_selected):
            best_selected = sel

    # Strategy 4: Local search refinement
    def local_search(selected_list, time_limit, max_rounds=500):
        selected = list(selected_list)
        sel_set = set(selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        for _ in range(max_rounds):
            if time.time() > time_limit:
                break

            # Try simple addition
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                continue

            # Try 1-for-2 swap
            found_swap = False
            if len(selected) > 30:
                trial_indices = np.random.choice(len(selected), size=min(20, len(selected)), replace=False)
            else:
                trial_indices = range(len(selected))

            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                mask_except_s[s] = 0
                mask_except_s[list(sel_set)] = 0

                candidates_for_swap = np.where(mask_except_s > 0)[0]
                if len(candidates_for_swap) < 2:
                    continue

                sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]
                found_pair = False
                for i in range(min(30, len(sorted_cands))):
                    c1 = sorted_cands[i]
                    compat_with_c1 = compat[c1]
                    for j in range(i + 1, min(60, len(sorted_cands))):
                        c2 = sorted_cands[j]
                        if compat_with_c1[c2]:
                            selected.remove(s)
                            sel_set.discard(s)
                            selected.append(c1)
                            sel_set.add(c1)
                            selected.append(c2)
                            sel_set.add(c2)
                            compat_with_all = np.ones(n, dtype=np.uint8)
                            for s3 in selected:
                                compat_with_all = compat_with_all & compat[s3]
                            for s3 in selected:
                                compat_with_all[s3] = 0
                            found_swap = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return selected

    if best_selected and time.time() < deadline - 5.0:
        improved = local_search(best_selected, deadline - 5.0, max_rounds=800)
        if len(improved) > len(best_selected):
            best_selected = improved

    # Final greedy fill
    if best_selected and time.time() < deadline - 2.0:
        sel_set = set(best_selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in best_selected:
            compat_with_all = compat_with_all & compat[s]
        for s in best_selected:
            compat_with_all[s] = 0
        addable = np.where(compat_with_all > 0)[0]
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 2.0:
                break
            best_selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
