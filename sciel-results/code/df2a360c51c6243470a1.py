"""QAP tai256c solver: vectorized 2-opt local search + iterated local search."""
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120s

    # Precompute row sums and column sums of a for faster delta computation
    # Also precompute b rows as a view

    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        return int(np.dot(a.ravel(), b[np.ix_(p, p)].ravel()))

    def swap_delta(p, u, v):
        """Compute delta cost when swapping p[u] and p[v]. O(n) vectorized."""
        pu = p[u]
        pv = p[v]

        # d_row[k] = b[pv, k] - b[pu, k] for all k
        d_row = b[pv, :] - b[pu, :]
        # d_col[k] = b[k, pv] - b[k, pu] for all k
        d_col = b[:, pv] - b[:, pu]

        # s1 = sum_{j not in {u,v}} (a[u,j] - a[v,j]) * d_row[p[j]]
        # = sum_j (a[u,j]-a[v,j]) * d_row[p[j]] - corrections
        s1 = np.dot(a[u, :] - a[v, :], d_row[p])
        s1 -= (a[u, u] - a[v, u]) * d_row[pv]
        s1 -= (a[u, v] - a[v, v]) * d_row[pu]

        # s2 = sum_{i not in {u,v}} (a[i,u] - a[i,v]) * d_col[p[i]]
        # = sum_i (a[i,u]-a[i,v]) * d_col[p[i]] - corrections
        s2 = np.dot(a[:, u] - a[:, v], d_col[p])
        s2 -= (a[u, u] - a[u, v]) * d_col[pv]
        s2 -= (a[v, u] - a[v, v]) * d_col[pu]

        # Diagonal and cross terms
        diag_cross = (a[u, u] * (b[pv, pv] - b[pu, pu])
                      + a[v, v] * (b[pu, pu] - b[pv, pv])
                      + a[u, v] * (b[pv, pu] - b[pu, pv])
                      + a[v, u] * (b[pu, pv] - b[pv, pu]))

        return s1 + s2 + diag_cross

    def local_search(p, cost, max_passes=100):
        """Vectorized 2-opt local search. Returns (p, cost)."""
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            # Vectorized: compute all swap deltas at once
            # For each pair (i, j), compute delta
            # This is O(n^2 * n) = O(n^3) per pass if done naively
            # But we can vectorize over j for each i
            
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                # Compute deltas for all j > i
                # We need to be careful with the vectorization
                # Let's compute for all j at once
                js = np.arange(i + 1, n)
                if len(js) == 0:
                    continue
                
                # For each j in js, compute swap_delta(p, i, j)
                # This requires accessing p[j] for each j
                # Let's do it in a loop but with numpy operations
                # Actually, let's just do the inner loop but with minimal overhead
                
                pi = p[i]
                # Precompute for this i
                au = a[i, :]
                a_u = a[:, i]
                
                for j in js:
                    pj = p[j]
                    if pi == pj:
                        continue
                    
                    d_row = b[pj, :] - b[pi, :]
                    d_col = b[:, pj] - b[:, pi]
                    
                    av = a[j, :]
                    a_v = a[:, j]
                    
                    s1 = np.dot(au - av, d_row[p])
                    s1 -= (au[i] - av[i]) * d_row[pj]
                    s1 -= (au[j] - av[j]) * d_row[pi]
                    
                    s2 = np.dot(a_u - a_v, d_col[p])
                    s2 -= (a_u[i] - a_v[i]) * d_col[pj]
                    s2 -= (a_u[j] - a_v[j]) * d_col[pi]
                    
                    diag_cross = (a[i, i] * (b[pj, pj] - b[pi, pi])
                                  + a[j, j] * (b[pi, pi] - b[pj, pj])
                                  + a[i, j] * (b[pj, pi] - b[pi, pj])
                                  + a[j, i] * (b[pi, pj] - b[pj, pi]))
                    
                    delta = s1 + s2 + diag_cross
                    
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += int(delta)
                        improved = True
                        # After swap, pi and pj change, so we need to restart
                        # Actually, for simplicity, let's just continue
                        # The next pass will pick up the changes
                        break  # Restart the inner loop for this i
        
        return p, cost

    # Better local search: fully vectorized over all pairs
    def local_search_fast(p, cost, max_passes=50):
        """Faster local search using numpy vectorization where possible."""
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                pi = p[i]
                au = a[i, :]
                a_u = a[:, i]
                
                # Compute deltas for all j > i in a vectorized manner
                # We need: for each j, delta(i, j)
                # Let's compute the key components
                
                for j in range(i + 1, n):
                    pj = p[j]
                    if pi == pj:
                        continue
                    
                    # d_row = b[pj,:] - b[pi,:]
                    # d_col = b[:,pj] - b[:,pi]
                    
                    # s1 = dot(au - a[j,:], d_row[p]) - corrections
                    # s2 = dot(a_u - a[:,j], d_col[p]) - corrections
                    
                    diff_row = au - a[j, :]
                    diff_col = a_u - a[:, j]
                    
                    d_row = b[pj, :] - b[pi, :]
                    d_col = b[:, pj] - b[:, pi]
                    
                    s1 = np.dot(diff_row, d_row[p])
                    s1 -= (au[i] - a[j, i]) * d_row[pj]
                    s1 -= (au[j] - a[j, j]) * d_row[pi]
                    
                    s2 = np.dot(diff_col, d_col[p])
                    s2 -= (a_u[i] - a[j, i]) * d_col[pj]
                    s2 -= (a_u[j] - a[j, j]) * d_col[pi]
                    
                    diag_cross = (a[i, i] * (b[pj, pj] - b[pi, pi])
                                  + a[j, j] * (b[pi, pi] - b[pj, pj])
                                  + a[i, j] * (b[pj, pi] - b[pi, pj])
                                  + a[j, i] * (b[pi, pj] - b[pj, pi]))
                    
                    delta = s1 + s2 + diag_cross
                    
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += int(delta)
                        improved = True
                        pi = p[i]  # Update pi after swap
                        au = a[i, :]
                        a_u = a[:, i]
                        # Don't break, continue checking other j's
        
        return p, cost

    # --- Initialization: multiple strategies ---
    rng = np.random.RandomState(42)
    
    # Strategy 1: Spectral initialization using Fiedler vectors
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Remove diagonal for Laplacian computation
    a_sym_off = a_sym.copy()
    b_sym_off = b_sym.copy()
    np.fill_diagonal(a_sym_off, 0)
    np.fill_diagonal(b_sym_off, 0)
    
    d_a = a_sym_off.sum(axis=1)
    L_a = np.diag(d_a) - a_sym_off
    
    d_b = b_sym_off.sum(axis=1)
    L_b = np.diag(d_b) - b_sym_off
    
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(L_a)
        eigvals_b, eigvecs_b = np.linalg.eigh(L_b)
    except Exception:
        eigvecs_a = np.eye(n)
        eigvecs_b = np.eye(n)
    
    # Fiedler vectors (second eigenvector)
    fiedler_a = eigvecs_a[:, 1]
    fiedler_b = eigvecs_b[:, 1]
    
    # Sort by Fiedler vector
    order_a = np.argsort(fiedler_a)
    order_b = np.argsort(fiedler_b)
    p_spectral = np.empty(n, dtype=np.int64)
    p_spectral[order_a] = order_b
    
    # Strategy 2: Rank matching by row sums
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    # Strategy 3: Random
    p_random = rng.permutation(n)
    
    # Strategy 4: Greedy nearest-neighbor in embedding space
    # Use first 3 eigenvectors for embedding
    X_a = eigvecs_a[:, 1:4]
    X_b = eigvecs_b[:, 1:4]
    
    # Greedy: for each facility in order of decreasing flow sum, assign to nearest unassigned location
    p_greedy = np.empty(n, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    facility_order = np.argsort(-a_row_sums)
    
    for fac in facility_order:
        # Find nearest unassigned location
        if np.all(assigned):
            break
        unassigned = np.where(~assigned)[0]
        if len(unassigned) == 0:
            break
        dists = np.sum((X_b[unassigned] - X_a[fac])**2, axis=1)
        best_loc = unassigned[np.argmin(dists)]
        p_greedy[fac] = best_loc
        assigned[best_loc] = True
    
    # Fill any unassigned (shouldn't happen)
    unassigned_facs = np.where(np.isnan(p_greedy.astype(float)))[0]
    unassigned_locs = np.where(~assigned)[0]
    for f, l in zip(unassigned_facs, unassigned_locs):
        p_greedy[f] = l
    
    # Evaluate all initializations
    candidates = [p_spectral.copy(), p_rank.copy(), p_random.copy(), p_greedy.copy()]
    best_p = None
    best_cost = np.inf
    
    for cand in candidates:
        c = compute_cost(cand)
        if c < best_cost:
            best_cost = c
            best_p = cand.copy()
    
    # Local search from best initialization
    best_p, best_cost = local_search_fast(best_p, best_cost, max_passes=100)
    
    # Iterated local search
    while time.time() < deadline - 3.0:
        # Perturb: random 2-8 swaps
        p = best_p.copy()
        n_swaps = rng.randint(2, 9)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        
        # Local search
        p, cost = local_search_fast(p, compute_cost(p), max_passes=50)
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
    
    return best_p.tolist()
