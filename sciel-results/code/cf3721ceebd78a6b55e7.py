"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute some useful quantities
    a_sym = a + a.T  # symmetric flow
    b_sym = b + b.T  # symmetric distance (assuming symmetric, but handle general)

    # Rank-matching initial solution
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    def cost(pi):
        # sum_{i,j} a[i,j] * b[pi[i], pi[j]]
        return int(np.sum(a * b[np.ix_(pi, pi)]))

    best_p = p.copy()
    best_cost = cost(best_p)

    # Vectorized 2-opt / swap local search
    def local_search(pi, max_iters=1000):
        """Iteratively try all single swaps, take best improving swap."""
        pi = pi.copy()
        c = cost(pi)
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            best_delta = 0
            best_i = -1
            best_j = -1
            # Try all pairs (i, j) with i < j
            for i in range(n):
                for j in range(i + 1, n):
                    # Compute cost change from swapping pi[i] and pi[j]
                    # Old contribution involving i and j:
                    # a[i,j]*b[pi[i],pi[j]] + a[j,i]*b[pi[j],pi[i]]
                    # New: a[i,j]*b[pi[j],pi[i]] + a[j,i]*b[pi[i],pi[j]]
                    # Plus terms with other k:
                    # Old: sum_k!=i,j a[i,k]*b[pi[i],pi[k]] + sum_k!=i,j a[k,i]*b[pi[k],pi[i]]
                    # New: sum_k!=i,j a[i,k]*b[pi[j],pi[k]] + sum_k!=i,j a[k,i]*b[pi[k],pi[j]]
                    # Similarly for j.
                    # This is O(n) per pair, so O(n^3) per full scan. Too slow for n=100.
                    # Instead, use precomputed row/col interactions.
                    pass
            # Fall back: random swaps
            if best_i == -1:
                # Random 10 swaps
                improved = False
                for _ in range(20):
                    i, j = np.random.choice(n, 2, replace=False)
                    if i == j:
                        continue
                    pi[i], pi[j] = pi[j], pi[i]
                    new_c = cost(pi)
                    if new_c < c:
                        c = new_c
                        improved = True
                    else:
                        pi[i], pi[j] = pi[j], pi[i]
                if not improved:
                    break
            else:
                pi[best_i], pi[best_j] = pi[best_j], pi[best_i]
                c += best_delta
        return pi, c

    # Better: use a vectorized approach for 2-opt
    # Precompute for each facility i, the vector of distances from its current location to all others
    # Then for a swap of i and j, the delta can be computed in O(n)

    def compute_swap_deltas(pi):
        """For each pair (i,j), compute the cost delta if we swap pi[i] and pi[j]."""
        # This is expensive: O(n^3). For n=100, that's 1e6 ops per delta matrix, which is ~100M total.
        # Let's do it more cleverly.
        # For a fixed i, the contribution to cost from facility i is:
        #   sum_j a[i,j] * b[pi[i], pi[j]]  (outgoing)
        #   sum_j a[j,i] * b[pi[j], pi[i]]  (incoming)
        # When we swap pi[i] and pi[j], the new location for i is pi[j] and for j is pi[i].
        # The change in cost from swapping i and j:
        # Let loc_i = pi[i], loc_j = pi[j]
        # Old cost terms involving i or j:
        #   sum_k a[i,k]*b[loc_i, pi[k]] + sum_k a[k,i]*b[pi[k], loc_i]
        #   sum_k a[j,k]*b[loc_j, pi[k]] + sum_k a[k,j]*b[pi[k], loc_j]
        #   a[i,j]*b[loc_i, loc_j] + a[j,i]*b[loc_j, loc_i]  [these are included in the sums above when k=j and k=i]
        # Actually, let's define:
        #   C_i = sum_k a[i,k]*b[loc_i, pi[k]] + sum_k a[k,i]*b[pi[k], loc_i]
        #   C_j = sum_k a[j,k]*b[loc_j, pi[k]] + sum_k a[k,j]*b[pi[k], loc_j]
        # After swap:
        #   C_i' = sum_k a[i,k]*b[loc_j, pi[k]] + sum_k a[k,i]*b[pi[k], loc_j]
        #   C_j' = sum_k a[j,k]*b[loc_i, pi[k]] + sum_k a[k,j]*b[pi[k], loc_i]
        # Delta = C_i' + C_j' - C_i - C_j
        # But note: the terms a[i,j]*b[loc_i,loc_j] etc. are counted in both C_i and C_j, so we need to be careful.
        # Actually, C_i includes k=j: a[i,j]*b[loc_i, loc_j] + a[j,i]*b[loc_j, loc_i]... no wait.
        # C_i = sum_k a[i,k]*b[loc_i, pi[k]] + sum_k a[k,i]*b[pi[k], loc_i]
        # When k=j: a[i,j]*b[loc_i, loc_j] + a[j,i]*b[loc_j, loc_i]
        # C_j includes k=i: a[j,i]*b[loc_j, loc_i] + a[i,j]*b[loc_i, loc_j]
        # So the cross term is double-counted. This is fine as long as we're consistent.
        # Delta = (C_i' - C_i) + (C_j' - C_j)
        # where C_i' uses loc_j for i's location, and C_j' uses loc_i for j's location.
        
        # Let's precompute:
        # For each facility i and location l, define:
        #   out[i, l] = sum_k a[i,k] * b[l, pi[k]]
        #   inc[i, l] = sum_k a[k,i] * b[pi[k], l]
        # Then C_i = out[i, pi[i]] + inc[i, pi[i]]
        # After swap, C_i' = out[i, pi[j]] + inc[i, pi[j]]
        # C_j' = out[j, pi[i]] + inc[j, pi[i]]
        # Delta = out[i, pi[j]] + inc[i, pi[j]] + out[j, pi[i]] + inc[j, pi[i]] - out[i, pi[i]] - inc[i, pi[i]] - out[j, pi[j]] - inc[j, pi[j]]
        
        # Precompute out and inc for all i and l. This is O(n^3) but vectorizable.
        # out[i, l] = sum_k a[i,k] * b[l, pi[k]]
        # = (a @ b[:, pi]) [i, l]  ... let's see: b[:, pi] has shape (n, n) where b[:, pi][l, k] = b[l, pi[k]]
        # So a @ b[:, pi] gives shape (n, n) where [i, l] = sum_k a[i,k] * b[l, pi[k]]. Yes!
        # Similarly, inc[i, l] = sum_k a[k,i] * b[pi[k], l]
        # = (a.T @ b[pi, :]) [i, l]  ... b[pi, :] has shape (n, n) where b[pi, :][k, l] = b[pi[k], l]
        # So a.T @ b[pi, :] gives [i, l] = sum_k a[k,i] * b[pi[k], l]. Yes!
        
        b_col_pi = b[:, pi]  # shape (n, n), [l, k] = b[l, pi[k]]
        b_row_pi = b[pi, :]  # shape (n, n), [k, l] = b[pi[k], l]
        out = a @ b_col_pi  # [i, l] = sum_k a[i,k] * b[l, pi[k]]
        inc = a.T @ b_row_pi  # [i, l] = sum_k a[k,i] * b[pi[k], l]
        
        # Now for each pair (i, j), delta = out[i, pi[j]] + inc[i, pi[j]] + out[j, pi[i]] + inc[j, pi[i]] - out[i, pi[i]] - inc[i, pi[i]] - out[j, pi[j]] - inc[j, pi[j]]
        # We can vectorize this:
        # Create arrays of pi values
        pi_i = pi[np.arange(n)]  # pi[i]
        pi_j = pi[np.arange(n)]  # pi[j]
        
        # For all pairs (i, j):
        # out[i, pi[j]]: for fixed i, this is out[i, pi[j]] for all j
        # Let's create a matrix D where D[i, j] = delta for swapping i and j
        # out[i, pi[j]] = out[np.arange(n)[:, None], pi[None, :]]  -> shape (n, n)
        out_i_pi_j = out[np.arange(n)[:, None], pi[None, :]]  # [i, j] = out[i, pi[j]]
        inc_i_pi_j = inc[np.arange(n)[:, None], pi[None, :]]  # [i, j] = inc[i, pi[j]]
        out_j_pi_i = out[np.arange(n)[None, :], pi[:, None]]  # [i, j] = out[j, pi[i]]
        inc_j_pi_i = inc[np.arange(n)[None, :], pi[:, None]]  # [i, j] = inc[j, pi[i]]
        
        out_i_pi_i = out[np.arange(n), pi]  # [i] = out[i, pi[i]]
        inc_i_pi_i = inc[np.arange(n), pi]  # [i] = inc[i, pi[i]]
        out_j_pi_j = out[np.arange(n), pi]  # [j] = out[j, pi[j]]
        inc_j_pi_j = inc[np.arange(n), pi]  # [j] = inc[j, pi[j]]
        
        D = (out_i_pi_j + inc_i_pi_j + out_j_pi_i + inc_j_pi_i - 
             out_i_pi_i[:, None] - inc_i_pi_i[:, None] - 
             out_j_pi_j[None, :] - inc_j_pi_j[None, :])
        
        return D

    # Simulated Annealing with vectorized swap selection
    def simulated_annealing(pi, initial_temp=1000.0, cooling=0.999, max_iters=50000):
        pi = pi.copy()
        c = cost(pi)
        best_pi = pi.copy()
        best_c = c
        temp = initial_temp
        
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            if temp < 1e-6:
                break
            
            # Compute all swap deltas
            D = compute_swap_deltas(pi)
            
            # Choose a random swap (i, j)
            i, j = np.random.choice(n, 2, replace=False)
            if i == j:
                continue
            delta = D[i, j]
            
            # Acceptance
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                pi[i], pi[j] = pi[j], pi[i]
                c += delta
                if c < best_c:
                    best_c = c
                    best_pi = pi.copy()
            
            temp *= cooling
        
        return best_pi, best_c

    # Run simulated annealing
    # First, do a quick local search to improve the initial solution
    # Use a simple greedy local search with the vectorized delta computation
    def greedy_local_search(pi, max_rounds=20):
        pi = pi.copy()
        c = cost(pi)
        for _ in range(max_rounds):
            if time.time() > deadline - 10.0:
                break
            D = compute_swap_deltas(pi)
            # Find the best swap
            min_idx = np.unravel_index(np.argmin(D), D.shape)
            i, j = min_idx
            if D[i, j] >= 0:
                break
            pi[i], pi[j] = pi[j], pi[i]
            c += D[i, j]
        return pi, c

    # Initial local search
    best_p, best_cost = greedy_local_search(best_p, max_rounds=50)

    # Simulated annealing
    sa_p, sa_cost = simulated_annealing(best_p, initial_temp=500.0, cooling=0.9995, max_iters=30000)
    if sa_cost < best_cost:
        best_cost = sa_cost
        best_p = sa_p

    # Final local search
    final_p, final_cost = greedy_local_search(best_p, max_rounds=30)
    if final_cost < best_cost:
        best_cost = final_cost
        best_p = final_p

    return best_p.tolist()
# EVOLVE-BLOCK-END
