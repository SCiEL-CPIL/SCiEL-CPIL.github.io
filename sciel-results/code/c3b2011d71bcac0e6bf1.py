"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def recompute_precomp(p):
        """
        Precompute dot_row and dot_col for O(1) swap delta.
        dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
        dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
        """
        b_col_p = b[:, p]  # shape (n, n): [l, k] = b[l, p[k]]
        b_row_p = b[p, :]  # shape (n, n): [k, l] = b[p[k], l]
        dot_row = a @ b_col_p.T  # (n, n): [i, l] = sum_k a[i,k]*b[l,p[k]]
        dot_col = a.T @ b_row_p  # (n, n): [i, l] = sum_k a[k,i]*b[p[k],l]
        return dot_row, dot_col

    def compute_swap_delta(i, j, p, dot_row, dot_col):
        """
        Compute cost delta when swapping facilities at indices i and j in permutation p.
        Uses precomputed dot_row and dot_col for O(1) computation.
        """
        u = p[i]
        v = p[j]
        
        if u == v:
            return 0.0

        buu = b[u, u]
        bvv = b[v, v]
        buv = b[u, v]
        bvu = b[v, u]

        delta = 0.0
        # Terms involving i and j directly
        delta += a[i, i] * (bvv - buu)
        delta += a[i, j] * (bvu - buv)
        delta += a[j, i] * (buv - bvu)
        delta += a[j, j] * (buu - bvv)
        
        # Terms with other facilities k (using precomputed dots)
        # sum_{k != i,j} a[i,k] * (b[v, p[k]] - b[u, p[k]])
        # = dot_row[i, v] - dot_row[i, u] - a[i,i]*(bvv-buu) - a[i,j]*(bvu-buv)
        delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(bvv - buu) - a[i,j]*(bvu - buv)
        
        # sum_{k != i,j} a[j,k] * (b[u, p[k]] - b[v, p[k]])
        # = dot_row[j, u] - dot_row[j, v] - a[j,i]*(buv-bvu) - a[j,j]*(buu-bvv)
        delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(buv - bvu) - a[j,j]*(buu - bvv)
        
        # sum_{k != i,j} a[k,i] * (b[p[k], v] - b[p[k], u])
        # = dot_col[i, v] - dot_col[i, u] - a[i,i]*(bvv-buu) - a[j,i]*(bvu-buv)
        delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(bvv - buu) - a[j,i]*(bvu - buv)
        
        # sum_{k != i,j} a[k,j] * (b[p[k], u] - b[p[k], v])
        # = dot_col[j, u] - dot_col[j, v] - a[i,j]*(buv-bvu) - a[j,j]*(buu-bvv)
        delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(buv - bvu) - a[j,j]*(buu - bvv)
        
        return delta

    def update_precomp(p, i, j, dot_row, dot_col):
        """
        Update dot_row and dot_col in-place after swapping p[i] and p[j].
        u = p[i] (old), v = p[j] (old). After swap, p[i]=v, p[j]=u.
        """
        u = p[i]
        v = p[j]
        
        # diff_row[l] = b[l, v] - b[l, u]
        # diff_col[l] = b[v, l] - b[u, l]
        diff_row = b[:, v] - b[:, u]
        diff_col = b[v, :] - b[u, :]
        
        # Update dot_row
        # For k in 0..n-1:
        # dot_row[k, l] changes by:
        # if k == i: a[i,i]*(b[l,v]-b[l,u]) + a[i,j]*(b[l,u]-b[l,v]) = (a[i,i]-a[i,j])*diff_row[l]
        # if k == j: a[j,i]*(b[l,v]-b[l,u]) + a[j,j]*(b[l,u]-b[l,v]) = (a[j,i]-a[j,j])*diff_row[l]
        # else: a[k,i]*(b[l,v]-b[l,u]) + a[k,j]*(b[l,u]-b[l,v]) = (a[k,i]-a[k,j])*diff_row[l]
        
        # Vectorized update for all k
        # delta_dot_row[k, l] = (a[k,i] - a[k,j]) * diff_row[l]
        # This is outer product of (a[:,i] - a[:,j]) and diff_row
        delta_dot_row = np.outer(a[:, i] - a[:, j], diff_row)
        dot_row += delta_dot_row
        
        # Update dot_col
        # dot_col[k, l] = sum_m a[m,k] * b[p[m], l]
        # Change in p: p[i] u->v, p[j] v->u
        # For m == i: a[i,k]*(b[v,l]-b[u,l]) = a[i,k]*diff_col[l]
        # For m == j: a[j,k]*(b[u,l]-b[v,l]) = -a[j,k]*diff_col[l]
        # For m != i,j: 0
        # So delta_dot_col[k, l] = (a[i,k] - a[j,k]) * diff_col[l]
        # Note: a[i,k] is element (i,k) of a. We need vector a[i,:] - a[j,:]
        delta_dot_col = np.outer(a[i, :] - a[j, :], diff_col)
        dot_col += delta_dot_col

    def local_search(p, cost, dot_row, dot_col, end_time, first_improvement=True):
        """
        Perform 2-swap Local Search.
        """
        p = p.copy()
        dot_row = dot_row.copy()
        dot_col = dot_col.copy()
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            
            # Iterate over all pairs. To avoid bias, we could shuffle, but for n=150,
            # full pass is 11175 pairs. O(1) delta makes this fast.
            # We'll do a standard full pass.
            for i in range(n):
                if time.time() > end_time:
                    return p, cost, dot_row, dot_col
                for j in range(i + 1, n):
                    delta = compute_swap_delta(i, j, p, dot_row, dot_col)
                    if delta < -1e-9:
                        # Accept move
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        update_precomp(p, i, j, dot_row, dot_col)
                        improved = True
                        if first_improvement:
                            # Restart the scan from the beginning to ensure we don't miss better moves
                            # near the current position due to the change in state
                            break
                if improved and first_improvement:
                    break
                    
        return p, cost, dot_row, dot_col

    def perturb(p, num_swaps, rng):
        """Perturb permutation by random swaps."""
        p_pert = p.copy()
        for _ in range(num_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        return p_pert

    def il_search(p, cost, dot_row, dot_col, end_time):
        """
        Iterated Local Search.
        """
        rng = np.random.RandomState(42)
        best_p = p.copy()
        best_cost = cost
        best_dot_row = dot_row.copy()
        best_dot_col = dot_col.copy()
        
        curr_p = p.copy()
        curr_cost = cost
        curr_dot_row = dot_row.copy()
        curr_dot_col = dot_col.copy()
        
        while time.time() < end_time:
            # Local search
            curr_p, curr_cost, curr_dot_row, curr_dot_col = local_search(
                curr_p, curr_cost, curr_dot_row, curr_dot_col, end_time, first_improvement=True
            )
            
            if curr_cost < best_cost:
                best_p = curr_p.copy()
                best_cost = curr_cost
                best_dot_row = curr_dot_row.copy()
                best_dot_col = curr_dot_col.copy()
            
            # Perturb
            # Adaptive perturbation size
            num_swaps = rng.randint(3, 10)
            curr_p = perturb(best_p, num_swaps, rng)
            curr_cost = compute_cost_fast(curr_p)
            curr_dot_row, curr_dot_col = recompute_precomp(curr_p)
            
        return best_p, best_cost

    # 1. Hungarian Initialization
    # Use linear assignment on a linearized QAP
    # A common linearization is C = a @ b.T or similar.
    # Let's try C = a @ b.T
    C = a @ b.T
    row_ind, col_ind = linear_sum_assignment(-C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    
    best_p = p_hung.copy()
    best_cost = compute_cost_fast(p_hung)
    best_dot_row, best_dot_col = recompute_precomp(best_p)
    
    # 2. Polish Hungarian solution with Local Search
    end_time_lk = deadline - 30.0
    p_lk, cost_lk, dot_lk_r, dot_lk_c = local_search(best_p, best_cost, best_dot_row, best_dot_col, end_time_lk, first_improvement=True)
    if cost_lk < best_cost:
        best_p = p_lk
        best_cost = cost_lk
        best_dot_row = dot_lk_r
        best_dot_col = dot_lk_c

    # 3. Iterated Local Search
    if time.time() < deadline - 5.0:
        p_ils, cost_ils = il_search(best_p, best_cost, best_dot_row, best_dot_col, deadline - 2.0)
        if cost_ils < best_cost:
            best_p = p_ils
            best_cost = cost_ils
            best_dot_row, best_dot_col = recompute_precomp(best_p)
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
