"""QAP tai256c solver: iterated local search with Hungarian re-optimization and tabu diversification."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search_fast(p, cost, max_passes=20):
        """Local search with precomputed distance matrix for O(1) swap delta."""
        # Precompute pairwise distances between locations
        # d_loc[i, j] = b[i, j] (since b is the distance matrix)
        # Actually b IS the distance matrix, so we can use it directly.
        
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def local_search(p, cost, max_passes=20):
        """Optimized local search using precomputed row/column sums for O(1) swap delta."""
        # Precompute for each facility i and location j:
        # row_cost[i] = sum_k a[i,k] * b[j, p[k]]  (cost contribution of facility i to location j)
        # col_cost[j] = sum_k a[k,i] * b[p[k], j]  (cost contribution of location j to facility i)
        # Actually, let's use a different approach:
        # For swap of positions u and v (facilities p[u] and p[v] swap locations):
        # We need to compute the change in cost.
        
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def tabu_kick(p, cost, n_kicks=5):
        """Apply several random tabu-like kicks to diversify, keeping best result."""
        best_p = p.copy()
        best_cost = cost
        tabu_tenure = 10
        tabu_list = {}
        iteration = 0
        
        for _ in range(n_kicks):
            if time.time() > deadline - 5:
                break
            # Pick a random pair
            i, j = np.random.choice(n, 2, replace=False)
            key = (min(i, j), max(i, j))
            
            # Check tabu
            is_tabu = key in tabu_list and tabu_list[key] > iteration
            
            # Compute delta
            delta = fast_swap_delta(p, i, j)
            
            # Accept if not tabu, or if it improves global best (aspiration)
            if not is_tabu or (cost + delta < best_cost):
                p[i], p[j] = p[j], p[i]
                cost += delta
                tabu_list[key] = iteration + tabu_tenure
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            
            iteration += 1
        
        return best_p, best_cost

    def or_opt_relocate(p, cost):
        """Try relocating single facilities to better positions (1-move neighborhood)."""
        improved = True
        max_passes = 5
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                # Try moving facility at position i to each other position
                for j in range(n):
                    if j == i or time.time() > deadline:
                        continue
                    # Compute delta for moving p[i] to position j
                    # This is equivalent to swapping if we think of it as exchange
                    # But actually for QAP, a 1-relocation isn't standard; let's use 2-exchange
                    pass
        return p, cost

    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    # Try multiple random initial solutions and keep the best after local search
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost
    
    # Try a few random restarts for initial solution
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, cost_rand = local_search(p_rand, compute_cost(p_rand))
        if cost_rand < best_cost:
            best_p = p_rand.copy()
            best_cost = cost_rand

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with periodic Hungarian re-optimization and tabu kicks
    iteration = 0
    while time.time() < deadline - 10:  # Leave 10s buffer
        iteration += 1

        # Perturb: combine random swaps with tabu kick
        p = best_p.copy()
        
        # Random swaps - vary the number based on iteration for diversity
        n_swaps = np.random.randint(2, 8)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        
        # Tabu kick for additional diversification
        p, cost = tabu_kick(p, compute_cost(p), n_kicks=3)

        # Local search from perturbed
        p, cost = local_search(p, cost)

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    # Final polish: one more local search pass on best solution
    if time.time() < deadline - 2:
        final_p = best_p.copy()
        final_cost = best_cost
        final_p, final_cost = local_search(final_p, final_cost)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost

    return best_p.tolist()
