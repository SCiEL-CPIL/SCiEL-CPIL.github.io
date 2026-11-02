import itertools
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

D = 11
DEADLINE = time.time() + 140  # safety margin before 150s kill


def _remaining_time():
    return max(0.0, DEADLINE - time.time())


def generate_pool():
    """Generate integer vectors with 2,3,4 non-zero entries (±1)."""
    vecs = []
    for w in (2, 3, 4):
        for combo in itertools.combinations(range(D), w):
            for signs in itertools.product((1, -1), repeat=w):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def build_compatibility(vectors, min_dist_sq):
    """
    Build adjacency: A[i,j] = 1 if vectors i,j are COMPATIBLE (can coexist).
    Compatible means ||vi - vj||^2 >= min_dist_sq.
    """
    n = len(vectors)
    # Compute all pairwise squared distances
    # Use: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    norms_sq = np.sum(vectors * vectors, axis=1)
    dots = vectors @ vectors.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    # Compatibility: dist_sq >= min_dist_sq (and i != j)
    compat = (dist_sq >= min_dist_sq)
    np.fill_diagonal(compat, False)
    return compat


def spectral_seed(compat, n_keep, rng):
    """
    Use spectral relaxation: find largest eigenvector of the Laplacian,
    keep top n_keep vertices by eigenvector magnitude.
    """
    n = len(compat)
    # Laplacian: L = D - A
    deg = np.sum(compat, axis=1).astype(float)
    L = np.diag(deg) - compat.astype(float)
    # Use scipy for eigendecomposition (Laplacian is symmetric)
    try:
        from scipy.linalg import eigh
        # Solve for smallest eigenvalues (Fiedler vector gives best partition)
        # Actually for max independent set, we want the eigenvector of A with largest eigenvalue
        # Let's use the adjacency matrix directly
        A = compat.astype(float)
        vals, vecs = eigh(A, subset_by_index=[n - 1, n - 1])
        v = vecs[:, -1]
    except Exception:
        v = rng.random(n)
    
    # Keep vertices with largest |v_i|
    order = np.argsort(-np.abs(v))
    return order[:n_keep]


def min_degree_greedy(compat, seed_order=None, time_limit=None):
    """
    Minimum-degree greedy: repeatedly pick the vertex with fewest compatible
    neighbors among remaining vertices. This is the "most constrained first"
    strategy which tends to produce larger independent sets.
    """
    n = len(compat)
    if seed_order is not None:
        # Start with the seed order as a priority, but still use min-degree
        # among remaining
        pass
    
    remaining = set(range(n))
    selected = []
    compat_copy = compat.copy()
    
    start = time.time()
    
    while remaining:
        if time_limit and (time.time() - start) > time_limit:
            break
        if _remaining_time() < 5:
            break
        
        # For each remaining vertex, count compatible neighbors in remaining
        # That's the "degree within remaining"
        best = None
        best_deg = float('inf')
        
        for v in remaining:
            # Count neighbors of v that are still remaining
            neigh = np.where(compat_copy[v] & np.array([x in remaining for x in range(n)]))[0]
            deg = len(neigh)
            if deg < best_deg:
                best_deg = deg
                best = v
        
        if best is None:
            break
        
        selected.append(best)
        remaining.discard(best)
        # Remove best from compat (it's selected, so its neighbors can't be selected)
        for u in remaining:
            if compat_copy[best, u]:
                # u is incompatible with best, so u cannot be selected
                # Actually wait: compat means they CAN coexist.
                # If compat[best,u] is True, they CAN coexist, so u is still available.
                # If compat[best,u] is False, they CANNOT coexist, so u must be removed.
                pass
        
        # Remove all vertices INCOMPATIBLE with best from remaining
        to_remove = set()
        for u in remaining:
            if not compat_copy[best, u]:
                to_remove.add(u)
        remaining -= to_remove
    
    return selected


def min_degree_greedy_fast(compat, time_limit=None):
    """
    Faster min-degree greedy using numpy.
    """
    n = len(compat)
    remaining_mask = np.ones(n, dtype=bool)
    selected = []
    
    start = time.time()
    
    while remaining_mask.any():
        if time_limit and (time.time() - start) > time_limit:
            break
        if _remaining_time() < 5:
            break
        
        # Degree of each remaining vertex within remaining set
        # deg[i] = sum of compat[i, j] for j in remaining
        deg = (compat * remaining_mask[None, :]).sum(axis=1)
        
        # Among remaining, find the one with minimum degree
        deg[~remaining_mask] = float('inf')
        best = int(np.argmin(deg))
        
        if not remaining_mask[best]:
            break
        
        selected.append(best)
        remaining_mask[best] = False
        
        # Remove all vertices incompatible with best
        # Incompatible = NOT compat[best, :]
        incompatible = ~compat[best, :]
        remaining_mask &= ~incompatible
    
    return selected


def local_search_swap(compat, selected, time_limit=None, max_rounds=50):
    """
    Swap-based local search: try to improve the independent set by
    swapping one selected vertex with one unselected vertex, or
    adding/removing vertices.
    """
    n = len(compat)
    selected_set = set(selected)
    unselected = set(range(n)) - selected_set
    
    improved = True
    rounds = 0
    
    start = time.time()
    
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        
        if time_limit and (time.time() - start) > time_limit:
            break
        if _remaining_time() < 5:
            break
        
        # Try: for each selected vertex s, try to remove it and add
        # multiple unselected vertices that become compatible
        # Simpler: try swapping one out, one in if it increases size
        # Or: try to add an unselected vertex that's compatible with all selected
        
        # First, try to add any compatible unselected vertex
        added = False
        for u in list(unselected):
            if all(compat[u, s] for s in selected_set):
                selected_set.add(u)
                unselected.discard(u)
                improved = True
                added = True
                break
        
        if added:
            continue
        
        # Try swaps: remove s, add u
        for s in list(selected_set):
            # Removing s frees up all vertices incompatible with s
            # that were previously blocked
            # Find unselected vertices that are compatible with (selected - {s})
            candidates = []
            for u in unselected:
                if all(compat[u, t] for t in selected_set if t != s):
                    candidates.append(u)
            
            if len(candidates) >= 2:
                # We can remove s and add 2+ vertices
                # Add as many as possible
                new_sel = set(selected_set) - {s}
                count = 0
                for u in candidates:
                    if all(compat[u, t] for t in new_sel):
                        new_sel.add(u)
                        count += 1
                if len(new_sel) > len(selected_set):
                    selected_set = new_sel
                    unselected = set(range(n)) - selected_set
                    improved = True
                    break
    
    return list(selected_set)


def construct():
    global DEADLINE
    DEADLINE = time.time() + 140
    
    pool = generate_pool()
    n = len(pool)
    
    best_set = np.empty((0, D), dtype=np.int64)
    
    # Try different min_dist_sq values
    # For weight-2 vectors: ||v||^2 = 2, so min_dist_sq = 2 means dist >= 2
    # For mixed: need to check
    # The constraint is: min pairwise dist >= max norm
    # If we use weight-2 vectors (norm^2=2), min_dist_sq should be 2
    # But if we mix weights, the max norm could be 4 (weight-2) or 9 (weight-3) etc.
    # Let's be safe: use min_dist_sq = 4 (so dist >= 2, and max norm^2 = 4 for weight-2)
    # Actually, let's think: if all vectors have norm^2 <= 4, and min dist^2 >= 4, that works.
    # Weight-2: norm^2 = 2. Weight-3: norm^2 = 3. Weight-4: norm^2 = 4.
    # So if we only use weight-2 and weight-3, max norm^2 = 3, need min_dist_sq >= 3.
    # If we include weight-4, max norm^2 = 4, need min_dist_sq >= 4.
    
    # Strategy: try multiple pools with different constraints
    configs = [
        (pool, 3),   # weight 2,3,4 but require dist^2 >= 3 (works if max norm^2 <= 3, i.e., no weight-4)
        (pool, 4),   # require dist^2 >= 4 (safe for all weights up to 4)
    ]
    
    # Filter pool for weight-2 and 3 only for the first config
    weights = np.sum(pool * pool, axis=1)
    pool_w23 = pool[weights <= 3]
    pool_w2 = pool[weights == 2]
    
    configs = [
        (pool_w2, 2),    # only weight-2, min_dist_sq=2
        (pool_w23, 3),   # weight-2,3, min_dist_sq=3
        (pool, 4),       # all, min_dist_sq=4
    ]
    
    for vecs, md in configs:
        if _remaining_time() < 10:
            break
        
        nv = len(vecs)
        if nv == 0:
            continue
        
        # Build compatibility matrix
        compat = build_compatibility(vecs, md)
        
        # Multiple strategies
        strategies = []
        
        # Strategy 1: min-degree greedy with random restarts
        rng = np.random.default_rng(42)
        
        for restart in range(5):
            if _remaining_time() < 8:
                break
            
            # Randomly permute the order to break symmetry
            perm = rng.permutation(nv)
            compat_perm = compat[np.ix_(perm, perm)]
            
            selected_idx = min_degree_greedy_fast(compat_perm, time_limit=5)
            
            if len(selected_idx) > len(best_set):
                best_set = vecs[perm[selected_idx]].copy()
        
        # Strategy 2: spectral seed + greedy + local search
        if _remaining_time() > 15:
            # Get a large seed set
            n_seed = min(nv, 300)
            seed_order = spectral_seed(compat, n_seed, rng)
            
            # Use the seed as initial selection, then add greedily
            selected = list(seed_order)
            selected_set = set(selected)
            
            # Add compatible vertices greedily
            remaining = set(range(nv)) - selected_set
            for v in list(remaining):
                if all(compat[v, s] for s in selected_set):
                    selected_set.add(v)
            
            # Local search
            selected_set = set(local_search_swap(compat, list(selected_set), time_limit=5, max_rounds=20))
            
            if len(selected_set) > len(best_set):
                best_set = vecs[list(selected_set)].copy()
        
        # Strategy 3: pure min-degree + local search
        if _remaining_time() > 10:
            selected_idx = min_degree_greedy_fast(compat, time_limit=5)
            selected_set = set(local_search_swap(compat, selected_idx, time_limit=5, max_rounds=20))
            
            if len(selected_set) > len(best_set):
                best_set = vecs[list(selected_set)].copy()
    
    # Verify the result
    if len(best_set) > 0:
        # Check min pairwise distance >= max norm
        norms_sq = np.sum(best_set * best_set, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Check all pairwise distances
        n_best = len(best_set)
        valid = True
        for i in range(n_best):
            for j in range(i + 1, n_best):
                diff = best_set[i] - best_set[j]
                d2 = np.sum(diff * diff)
                if d2 < max_norm_sq:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback: return empty or a valid subset
            # Try to find a valid subset by removing violating pairs
            # Simple: just return first vector
            if n_best > 0:
                best_set = best_set[:1]
    
    return best_set


if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} vectors")
    if len(result) > 0:
        norms_sq = np.sum(result * result, axis=1)
        print(f"Max norm^2: {np.max(norms_sq)}")
        # Check min distance
        min_d2 = float('inf')
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.sum((result[i] - result[j]) ** 2)
                min_d2 = min(min_d2, d2)
        print(f"Min dist^2: {min_d2}")
        print(f"Valid: {min_d2 >= np.max(norms_sq)}")
