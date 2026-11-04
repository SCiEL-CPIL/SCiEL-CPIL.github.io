"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


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
        
        def calc_delta(i, j, perm):
            u = perm[i]
            v = perm[j]
            
            # Diagonal part
            delta = (a_diag[i] - a_diag[j]) * (b_diag[v] - b_diag[u])
            
            # Off-diagonal part
            Ai = a[i]
            Aj = a[j]
            Bv = b[v, perm]
            Bu = b[u, perm]
            
            diff_A = Ai - Aj
            diff_B = Bv - Bu
            
            total = np.dot(diff_A, diff_B)
            
            term_i = (Ai[i] - Aj[i]) * (Bv[i] - Bu[i])
            term_j = (Ai[j] - Aj[j]) * (Bv[j] - Bu[j])
            
            off_diag_sum = total - term_i - term_j
            delta += 2 * off_diag_sum
            
            return int(delta)

        # Iterated Local Search (Steepest Descent + Perturbation)
        # We use Steepest Descent to find a local optimum, then perturb.
        
        while time.time() < time_limit - 5.0:
            # Local Search: Steepest Descent
            improved = True
            while improved and time.time() < time_limit - 5.0:
                improved = False
                best_delta = 0
                best_move = None
                
                # Evaluate all moves
                for i in range(n):
                    u = current_p[i]
                    for j in range(i+1, n):
                        if time.time() > time_limit - 5.0:
                            break
                        delta = calc_delta(i, j, current_p)
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, j)
                    if time.time() > time_limit - 5.0:
                        break
                
                if best_move is None:
                    break
                
                i, j = best_move
                v = current_p[j]
                u = current_p[i]
                current_p[i], current_p[j] = v, u
                current_cost += best_delta
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
            
            # Perturbation
            # Perform a few random swaps to escape local optimum
            num_perturbs = random.randint(5, 15)
            for _ in range(num_perturbs):
                if time.time() > time_limit - 5.0:
                    break
                i_kick = random.randint(0, n-1)
                j_kick = random.randint(0, n-1)
                if i_kick != j_kick:
                    delta = calc_delta(i_kick, j_kick, current_p)
                    v = current_p[j_kick]
                    current_p[i_kick], current_p[j_kick] = v, current_p[i_kick]
                    current_cost += delta
                    
                    # If the perturbation improves the best, update it
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()

    else:
        # Fallback for non-symmetric (unlikely for tai256c)
        # Simple random search or first improvement
        while time.time() < time_limit - 5.0:
            i_kick = random.randint(0, n-1)
            j_kick = random.randint(0, n-1)
            if i_kick != j_kick:
                # Calculate cost change directly for non-symmetric
                u = current_p[i_kick]
                v = current_p[j_kick]
                # Swap
                current_p[i_kick], current_p[j_kick] = v, u
                new_cost = calculate_cost(current_p)
                if new_cost < current_cost:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_p = current_p.copy()
                else:
                    # Revert
                    current_p[i_kick], current_p[j_kick] = u, v

    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
