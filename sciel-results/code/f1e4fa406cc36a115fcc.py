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
    deadline = 115.0  # Safety margin
    
    # Symmetrize matrices to ensure cost is well-defined and simplifies updates
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Precompute initial cost
    def get_cost(p):
        return np.sum(a_sym * b_sym[np.ix_(p, p)])
    
    def delta_cost_swap(p, i, j, a_sym, b_sym):
        """
        Calculate cost change for swapping p[i] and p[j].
        Uses vectorized operation for O(n) speed.
        """
        u = p[i]
        v = p[j]
        
        d_a = a_sym[i, :] - a_sym[j, :]
        d_b = b_sym[v, :] - b_sym[u, :]
        
        s = np.dot(d_a, d_b[p])
        
        term_i = d_a[i] * d_b[u]
        term_j = d_a[j] * d_b[v]
        
        sum_k = s - term_i - term_j
        
        return 2 * sum_k

    def local_search(p, current_cost, max_passes=10):
        """First-improvement 2-opt local search."""
        improved = True
        ls_passes = 0
        while improved and ls_passes < max_passes:
            improved = False
            ls_passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, current_cost
                for j in range(i + 1, n):
                    dc = delta_cost_swap(p, i, j, a_sym, b_sym)
                    if dc < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        current_cost += dc
                        improved = True
            if time.time() > deadline:
                return p, current_cost
        return p, current_cost

    def spectral_init():
        """Spectral initialization using eigendecomposition."""
        # Use SVD of flow matrix for spectral assignment
        U, _, Vt = np.linalg.svd(a_sym, full_matrices=False)
        # Match top eigenvectors
        p = np.empty(n, dtype=np.int64)
        # Sort by eigenvector components
        for i in range(n):
            # Assign facility i to location with highest correlation
            # This is a simplified spectral approach
            pass
        
        # Fallback: rank matching
        heavy = np.argsort(-a_sym.sum(axis=1), kind="stable")
        central = np.argsort(b_sym.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # 1. Initialization: Rank matching
    heavy = np.argsort(-a_sym.sum(axis=1), kind="stable")
    central = np.argsort(b_sym.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = get_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # 2. Local Search
    p, current_cost = local_search(p, current_cost, max_passes=10)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. Simulated Annealing with restarts
    # Multiple restarts with different initial solutions
    
    # First, try SA from the current best
    T_start = 50.0
    alpha = 0.995
    min_T = 1e-3
    sa_passes = 0
    
    while time.time() < deadline:
        # Pick random i, j
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        dc = delta_cost_swap(p, i, j, a_sym, b_sym)
        
        if dc < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += dc
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            if T_start > min_T:
                if random.random() < np.exp(-dc / T_start):
                    p[i], p[j] = p[j], p[i]
                    current_cost += dc
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
        T_start *= alpha
        sa_passes += 1
        
        # Every 10000 steps, do a quick local search
        if sa_passes % 10000 == 0:
            if time.time() > deadline - 5:
                break
            p, current_cost = local_search(p, current_cost, max_passes=3)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            # Reset temperature for continued exploration
            if T_start < 5.0:
                T_start = 20.0
        
        if time.time() > deadline:
            break
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
