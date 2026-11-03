# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        # d[x] = b[lj, p[x]] - b[li, p[x]]
        # We want sum_x (a[i,x] - a[j,x] + a[x,i] - a[x,j]) * (b[lj, p[x]] - b[li, p[x]])
        # Note: terms where x=i or x=j are 0 in the difference part if handled correctly, 
        # but here we compute full vector diff.
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def two_opt_polish(p, max_time=None):
        """Greedy 2-opt local search. Returns (p, cost)."""
        if max_time is None:
            max_time = deadline
        cost = calc_cost_full(p)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Randomize order of pairs to avoid bias? 
            # Standard first-improvement is fast.
            # Let's stick to first-improvement for speed.
            for i in range(n):
                if time.time() > max_time:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        return p, cost
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def spectral_init():
        """
        Spectral bisection based initialization.
        Uses the structure of the distance matrix b.
        """
        # Compute Laplacian of b
        # L = D - B
        # D is diagonal of row sums of B
        # We want to partition locations into two sets with minimal cut.
        # Fiedler vector (eigenvector of L with 2nd smallest eigenvalue)
        # For n=256, eigendecomposition is feasible.
        
        # Symmetrize b just in case, though QAP instances are usually symmetric
        b_sym = (b + b.T) / 2.0
        d = b_sym.sum(axis=1)
        L = np.diag(d) - b_sym
        
        # Get eigenvalues and eigenvectors
        # We only need the 2nd smallest
        # scipy.linalg.eigh is efficient for symmetric
        try:
            from scipy.linalg import eigh
            vals, vecs = eigh(L, eigvals=(1, 2)) # 0-indexed, so 1 and 2 are 2nd and 3rd? 
            # eigh(a, eigvals=(k, k)) returns k-th eigenvalue.
            # Smallest is 0. Next is 1.
            # Let's just get all if n is small, or use subset.
            vals, vecs = eigh(L)
            # Find index of 2nd smallest eigenvalue
            # Usually 0 is the trivial one (all 1s).
            idx = 1
            fiedler = vecs[:, idx]
        except Exception:
            # Fallback to random if spectral fails
            return np.random.permutation(n).astype(np.int64)
        
        # Sort locations by Fiedler vector
        loc_order = np.argsort(fiedler, kind='stable')
        
        # Sort facilities by importance (row sum of a)
        fac_order = np.argsort(a.sum(axis=1), kind='stable')
        
        # Assign: most important facilities to most "extreme" locations?
        # Or just map order to order.
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        return p

    # --- Phase 1: Initial Solutions ---
    best_p = None
    best_cost = float('inf')
    
    # 1. Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy_order] = central_order
    p_rank, c_rank = two_opt_polish(p_rank, max_time=start_time + 5.0)
    if c_rank < best_cost:
        best_cost = c_rank
        best_p = p_rank.copy()

    # 2. Spectral Init
    if time.time() < start_time + 10.0:
        p_spec = spectral_init()
        p_spec, c_spec = two_opt_polish(p_spec, max_time=start_time + 15.0)
        if c_spec < best_cost:
            best_cost = c_spec
            best_p = p_spec.copy()

    # 3. Random Init
    if time.time() < start_time + 20.0:
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, c_rand = two_opt_polish(p_rand, max_time=start_time + 25.0)
        if c_rand < best_cost:
            best_cost = c_rand
            best_p = p_rand.copy()

    # --- Phase 2: Iterated Local Search (ILS) ---
    # Perturb, Polish, Keep Best
    current_p = best_p.copy()
    current_cost = best_cost
    
    # Parameters for ILS
    # Perturbation strength: number of swaps
    # Start with small, maybe increase if stuck?
    # Or just fixed random.
    
    while time.time() < deadline - 5.0:
        # Perturb: Random swaps
        # Number of swaps: proportional to n
        num_swaps = max(5, n // 20) 
        p_iter = current_p.copy()
        
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_iter[i], p_iter[j] = p_iter[j], p_iter[i]
                
        # Polish
        p_iter, c_iter = two_opt_polish(p_iter, max_time=deadline - 2.0)
        
        if c_iter < best_cost:
            best_cost = c_iter
            best_p = p_iter.copy()
            current_p = p_iter.copy()
            current_cost = c_iter
        else:
            # If not better, we might want to accept worse solutions to escape basins?
            # ILS usually just restarts from best or best + perturbation.
            # Let's stick to restarting from best_p to ensure we don't drift too far.
            # But if we always restart from best, we might get stuck in the same basin.
            # So, we perturb best_p each time.
            pass

    # --- Phase 3: Final Polish ---
    if time.time() < deadline - 1.0:
        p_final = best_p.copy()
        p_final, c_final = two_opt_polish(p_final, max_time=deadline - 0.5)
        if c_final < best_cost:
            best_p = p_final
            best_cost = c_final

    return best_p.tolist()

# EVOLVE-BLOCK-END
