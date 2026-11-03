"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    def compute_cost(p):
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    # Initial solution: Hungarian algorithm on linear surrogate
    # Surrogate: facility i has flow centrality a.sum(axis=1), location j has distance centrality b.sum(axis=1)
    # We want to assign high-flow facilities to low-distance locations
    # Cost matrix for Hungarian: C[i,j] = a.sum(axis=1)[i] * b.sum(axis=1)[j]
    # Minimizing this puts high flow * low distance together
    flow_cent = a.sum(axis=1)
    dist_cent = b.sum(axis=1)
    surrogate_cost = np.outer(flow_cent, dist_cent)
    # Hungarian minimizes sum of assigned costs
    row_ind, col_ind = linear_sum_assignment(surrogate_cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    best_p = p.copy()
    best_cost = compute_cost(p)

    def compute_delta_swap(p, i, j):
        u = p[i]
        v = p[j]
        # delta = sum_{k!=i,j} (a[i,k] - a[j,k])*(b[v,p[k]] - b[u,p[k]])
        #       + sum_{k!=i,j} (a[k,i] - a[k,j])*(b[p[k],v] - b[p[k],u])
        #       + (a[i,j] - a[j,i])*(b[v,u] - b[u,v])
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False

        diff_a_row = a[i, mask] - a[j, mask]
        diff_b_row = b[v, p[mask]] - b[u, p[mask]]
        term1 = np.dot(diff_a_row, diff_b_row)

        diff_a_col = a[mask, i] - a[mask, j]
        diff_b_col = b[p[mask], v] - b[p[mask], u]
        term2 = np.dot(diff_a_col, diff_b_col)

        term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])

        return term1 + term2 + term3

    # Simulated Annealing with vectorized swap evaluation
    np.random.seed(42)
    
    # SA parameters
    T_start = max(best_cost * 0.01, 1000.0)
    T_end = max(best_cost * 0.0001, 1.0)
    cooling_rate = 0.999
    max_iterations = 50000
    
    p = best_p.copy()
    cur_cost = best_cost
    
    # Precompute all pair indices for vectorized evaluation
    pair_indices = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    n_pairs = len(pair_indices)
    
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        if time.time() > deadline:
            break
            
        # Pick a random pair of facilities to swap
        idx = np.random.randint(0, n_pairs)
        i, j = pair_indices[idx]
        
        # Compute delta for this swap
        delta = compute_delta_swap(p, i, j)
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improving moves
            p[i], p[j] = p[j], p[i]
            cur_cost += delta
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_p = p.copy()
        else:
            # Accept worsening moves with probability exp(-delta/T)
            if np.random.random() < np.exp(-delta / T_start):
                p[i], p[j] = p[j], p[i]
                cur_cost += delta
        
        # Cool down
        T_start = max(T_start * cooling_rate, T_end)
        iteration += 1
        
        # Periodic local search polish
        if iteration % 500 == 0 and time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for pi in range(n):
                    if time.time() > deadline:
                        break
                    for pj in range(pi + 1, n):
                        d = compute_delta_swap(p, pi, pj)
                        if d < 0:
                            p[pi], p[pj] = p[pj], p[pi]
                            cur_cost += d
                            improved = True
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_p = p.copy()
    
    # Final polish: exhaustive local search
    if time.time() < deadline:
        p = best_p.copy()
        cur_cost = best_cost
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    delta = compute_delta_swap(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cur_cost += delta
                        improved = True
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
