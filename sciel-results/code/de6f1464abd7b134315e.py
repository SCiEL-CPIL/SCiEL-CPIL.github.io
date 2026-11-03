"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    """
    Solves the QAP instance wil100.
    a: flow matrix (n, n)
    b: distance matrix (n, n)
    Returns: permutation p of 0..n-1
    """
    n = len(a)
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Symmetrize matrices to ensure cost is well-defined and simplifies updates
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # 1. Initialization: Rank matching
    heavy = np.argsort(-a_sym.sum(axis=1), kind="stable")
    central = np.argsort(b_sym.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute initial cost
    def get_cost(p):
        return np.sum(a_sym * b_sym[np.ix_(p, p)])
    
    current_cost = get_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # 2. Quick Local Search (2-opt) to polish initial solution
    # We do a few passes of best-improvement or first-improvement 2-opt
    # Since n=100, O(n^2) moves per pass is 10,000 moves. Each move O(n) cost update.
    # 10,000 * 100 = 1,000,000 ops per pass. Very fast.
    
    def delta_cost_swap(p, i, j, a_sym, b_sym):
        """
        Calculate cost change for swapping p[i] and p[j].
        Uses vectorized operation for O(n) speed.
        """
        u = p[i]
        v = p[j]
        
        # d_a[k] = a[i,k] - a[j,k]
        # d_b[l] = b[v,l] - b[u,l]
        # Sum over k != i, j of d_a[k] * d_b[p[k]]
        
        d_a = a_sym[i, :] - a_sym[j, :]
        d_b = b_sym[v, :] - b_sym[u, :]
        
        # Vectorized dot product with permuted d_b
        # We need sum_{k} d_a[k] * d_b[p[k]]
        # This is equivalent to d_a @ d_b[p]
        s = np.dot(d_a, d_b[p])
        
        # Subtract terms for k=i and k=j
        # Term for k=i: d_a[i] * d_b[p[i]] = d_a[i] * d_b[u]
        # Term for k=j: d_a[j] * d_b[p[j]] = d_a[j] * d_b[v]
        term_i = d_a[i] * d_b[u]
        term_j = d_a[j] * d_b[v]
        
        sum_k = s - term_i - term_j
        
        # Total delta is 2 * sum_k
        # Note: The diagonal terms a[i,i] etc are handled correctly by the exclusion
        # The formula 2 * sum_{k!=i,j} (a_ik - a_jk)(b_vp(k) - b_up(k)) is standard for symmetric matrices
        return 2 * sum_k

    # Local Search
    improved = True
    ls_passes = 0
    while improved and ls_passes < 5:
        improved = False
        ls_passes += 1
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                dc = delta_cost_swap(p, i, j, a_sym, b_sym)
                if dc < -1e-9:
                    # Swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += dc
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
            if time.time() > deadline:
                break
    
    # 3. Simulated Annealing
    # Start temperature
    T = 100.0
    alpha = 0.95
    min_T = 1e-4
    
    # We can run SA until deadline
    while T > min_T and time.time() < deadline:
        # Pick random i, j
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        dc = delta_cost_swap(p, i, j, a_sym, b_sym)
        
        if dc < 0:
            # Always accept
            p[i], p[j] = p[j], p[i]
            current_cost += dc
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Accept with probability exp(-dc / T)
            if random.random() < np.exp(-dc / T):
                p[i], p[j] = p[j], p[i]
                current_cost += dc
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        T *= alpha
        
        # Check time periodically
        if time.time() > deadline:
            break

    return best_p.tolist()

# EVOLVE-BLOCK-END
