"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    
    # Work with int64 arrays for exact cost computation
    A = a.astype(np.int64)
    B = b.astype(np.int64)
    
    def compute_cost(p):
        """Compute the exact QAP cost for a permutation p."""
        # Cost = sum_{i,j} A[i,j] * B[p[i], p[j]]
        # Vectorized: for each i, sum_j A[i,j] * B[p[i], p[j]]
        total = 0
        for i in range(n):
            total += np.dot(A[i], B[p[i], p])
        return total
    
    def swap_delta(i, j, p):
        """Compute the change in cost when swapping p[i] and p[j]."""
        pi, pj = p[i], p[j]
        
        # For k != i, j:
        # delta_k = A[i,k]*(B[pj,pk]-B[pi,pk]) + A[k,i]*(B[pk,pj]-B[pk,pi])
        #         + A[j,k]*(B[pi,pk]-B[pj,pk]) + A[k,j]*(B[pk,pi]-B[pk,pj])
        # For the i,j pair itself:
        # delta_ij = A[i,j]*(B[pj,pi]-B[pi,pj]) + A[j,i]*(B[pi,pj]-B[pj,pi])
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        k_idx = np.where(mask)[0]
        
        delta = 0
        if len(k_idx) > 0:
            pk = p[k_idx]
            # A[i,k] terms: A[i,k] * (B[pj,pk] - B[pi,pk])
            delta += np.sum(A[i, k_idx] * (B[pj, pk] - B[pi, pk]))
            # A[k,i] terms: A[k,i] * (B[pk,pj] - B[pk,pi])
            delta += np.sum(A[k_idx, i] * (B[pk, pj] - B[pk, pi]))
            # A[j,k] terms: A[j,k] * (B[pi,pk] - B[pj,pk])
            delta += np.sum(A[j, k_idx] * (B[pi, pk] - B[pj, pk]))
            # A[k,j] terms: A[k,j] * (B[pk,pi] - B[pk,pj])
            delta += np.sum(A[k_idx, j] * (B[pk, pi] - B[pk, pj]))
        
        # Direct i-j and j-i terms
        delta += A[i, j] * (B[pj, pi] - B[pi, pj])
        delta += A[j, i] * (B[pi, pj] - B[pj, pi])
        
        return delta
    
    def local_search(p, current_cost, time_limit):
        """First-improvement local search with swap moves."""
        start = time.time()
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            for i in range(n):
                if (time.time() - start) > time_limit:
                    break
                for j in range(i + 1, n):
                    if (time.time() - start) > time_limit:
                        break
                    d = swap_delta(i, j, p)
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += d
                        improved = True
        return p, current_cost
    
    # === Multi-heuristic warm start ===
    np.random.seed(42)
    
    # Heuristic 1: Sort by row sums (heavy facilities to central locations)
    a_row_sums = A.sum(axis=1)
    b_row_sums = B.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    cost1 = compute_cost(p1)
    
    # Heuristic 2: Reverse sort (heavy facilities to peripheral locations)
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central[::-1]
    cost2 = compute_cost(p2)
    
    # Heuristic 3: Random permutation
    p3 = np.random.permutation(n)
    cost3 = compute_cost(p3)
    
    # Heuristic 4: Greedy assignment using Hungarian on a derived cost matrix
    # Approximate QAP by linear assignment: minimize sum_i A[i,:] . B[:,p[i]]
    # Use row-wise sums of A weighted by B
    # For each facility i, the cost of assigning to location j is sum_k A[i,k] * B[j, ?]
    # This is hard to make linear, so use a simpler heuristic:
    # Assign based on the product of row sums
    a_rs = a_row_sums.astype(np.float64)
    b_rs = b_row_sums.astype(np.float64)
    # Create a cost matrix where cost[i,j] = a_rs[i] * b_rs[j]
    cost_mat = np.outer(a_rs, b_rs)
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    p4 = np.zeros(n, dtype=np.int64)
    p4[row_ind] = col_ind
    cost4 = compute_cost(p4)
    
    # Heuristic 5: Another random
    np.random.seed(123)
    p5 = np.random.permutation(n)
    cost5 = compute_cost(p5)
    
    # Heuristic 6: Sort by column sums of A vs row sums of B
    a_col_sums = A.sum(axis=0)
    heavy_col = np.argsort(-a_col_sums, kind="stable")
    p6 = np.empty(n, dtype=np.int64)
    p6[heavy_col] = central
    cost6 = compute_cost(p6)
    
    # Pick the best initial solution
    candidates = [
        (p1, cost1),
        (p2, cost2),
        (p3, cost3),
        (p4, cost4),
        (p5, cost5),
        (p6, cost6),
    ]
    candidates.sort(key=lambda x: x[1])
    p, current_cost = candidates[0]
    
    # === Local search with multiple restarts ===
    start_time = time.time()
    time_budget = 115.0  # Leave 5 seconds margin
    
    # First pass: aggressive local search on the best initial solution
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5.0:
        p, current_cost = local_search(p, current_cost, remaining - 5.0)
    
    # Try perturbations and re-optimization
    np.random.seed(456)
    best_p = p.copy()
    best_cost = current_cost
    
    for restart in range(10):
        if time.time() - start_time > time_budget - 5.0:
            break
        
        # Create a perturbed version
        p_restart = best_p.copy()
        num_swaps = np.random.randint(1, 6)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
        
        cost_restart = compute_cost(p_restart)
        
        # Local search on the perturbed solution
        remaining = time_budget - (time.time() - start_time)
        if remaining > 5.0:
            p_restart, cost_restart = local_search(p_restart, cost_restart, remaining - 5.0)
        
        if cost_restart < best_cost:
            best_p = p_restart.copy()
            best_cost = cost_restart
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
