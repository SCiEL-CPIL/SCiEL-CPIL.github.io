"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Initial Solution: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    def calculate_cost(perm):
        # perm is a permutation of 0..n-1
        # Cost = sum a[i, j] * b[perm[i], perm[j]]
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    best_p = p.copy()
    best_cost = calculate_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    if a_sym and b_sym:
        a_diag = np.diag(a)
        b_diag = np.diag(b)
        
        # Precompute for fast cost update if needed, but we will use full Hungarian steps
        
        def get_hungarian_perm(perm):
            # Calculate cost matrix M for Hungarian algorithm
            # M[i, k] = cost of assigning facility i to location k, 
            # given other facilities are at their current positions in perm.
            
            # B_pi[j, k] = b[perm[j], k]
            B_pi = b[perm, :]
            
            # Prod[i, k] = sum_j a[i, j] * b[perm[j], k]
            Prod = a @ B_pi
            
            # M[i, k] = 2 * sum_{j != i} a[i, j] * b[k, perm[j]] + a[i, i] * b[k, k]
            # sum_{j != i} a[i, j] * b[k, perm[j]] = Prod[i, k] - a[i, i] * b[k, perm[i]]
            
            # M = 2 * (Prod - self_contrib) + new_self
            # self_contrib[i, k] = a[i, i] * b[k, perm[i]]
            # Note: b[k, perm[i]] is the element at row k, col perm[i] of b.
            # This is b.T[perm[i], k] or just indexing b.
            
            # Vectorized calculation:
            # self_contrib[i, k] = a_diag[i] * b[k, perm[i]]
            # We need a matrix S where S[i, k] = a_diag[i] * b[k, perm[i]]
            # b[:, perm] gives a matrix where col i is b[:, perm[i]]
            # So S = a_diag[:, None] * b[:, perm]
            
            S = a_diag[:, None] * b[:, perm]
            
            M = 2 * (Prod - S)
            
            # Add new self term: a[i, i] * b[k, k]
            # new_self[i, k] = a_diag[i] * b_diag[k]
            M += a_diag[:, None] * b_diag[None, :]
            
            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(M)
            return col_ind

        # Main Loop: Iterated Hungarian with Perturbation
        stagnation_count = 0
        max_stagnation = 10 # If no improvement in 10 Hungarian steps, perturb
        
        while time.time() < time_limit - 5.0:
            # 1. Hungarian Step
            new_p = get_hungarian_perm(current_p)
            
            # Check if it's a valid permutation (it should be)
            if len(new_p) != n:
                break
                
            new_cost = calculate_cost(new_p)
            
            if new_cost < current_cost:
                current_p = new_p
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
                # If stagnant, perturb to escape local optimum
                if stagnation_count >= max_stagnation:
                    # Random swap
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i != j:
                        current_p[i], current_p[j] = current_p[j], current_p[i]
                        # Update current_cost? It's better to recalculate or update incrementally.
                        # Recalculating is O(n^2), which is fast (65k ops).
                        current_cost = calculate_cost(current_p)
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
                    stagnation_count = 0
                    max_stagnation = random.randint(5, 20) # Reset stagnation threshold

            # Optional: Quick Polish?
            # Hungarian is expensive. Let's see how many iterations we get.
            # n=256. Hungarian is ~50ms. 115s / 50ms = 2300 iterations.
            # This is a lot. We might not need explicit 2-opt if Hungarian is exploring well.
            # However, Hungarian can cycle. Perturbation helps.

    else:
        # Fallback for non-symmetric (unlikely for tai256c)
        # Simple 2-opt
        while time.time() < time_limit - 5.0:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    # Swap and check
                    current_p[i], current_p[j] = current_p[j], current_p[i]
                    new_cost = calculate_cost(current_p)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
                        improved = True
                    else:
                        current_p[i], current_p[j] = current_p[j], current_p[i]
            if not improved:
                break

    # Final Polish with 2-opt on the best solution found
    if time.time() < time_limit - 2.0:
        p_polish = best_p.copy()
        cost_polish = best_cost
        improved = True
        while improved and time.time() < time_limit - 1.0:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    p_polish[i], p_polish[j] = p_polish[j], p_polish[i]
                    new_cost = calculate_cost(p_polish)
                    if new_cost < cost_polish:
                        cost_polish = new_cost
                        improved = True
                        if cost_polish < best_cost:
                            best_cost = cost_polish
                            best_p = p_polish.copy()
                    else:
                        p_polish[i], p_polish[j] = p_polish[j], p_polish[i]

    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
