import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Check if B is symmetric
    is_sym = np.allclose(b, b.T)
    
    # Initial assignment: Rank matching
    # Heavy flows (high row sum in A) to central locations (low row sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by decreasing flow sum
    heavy_order = np.argsort(-a_sums, kind='stable')
    # Sort locations by increasing distance sum (central)
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Calculate initial cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # Phase 1: Frank-Wolfe on the Birkhoff Polytope (Continuous Relaxation)
    # This phase uses a linear assignment step (Hungarian) to project the
    # continuous gradient onto the discrete permutation space, allowing
    # large structural jumps that escape local minima of simple swap methods.
    
    fw_deadline = start_time + 40.0
    
    # Precompute A_sym if B is symmetric for faster C construction
    if is_sym:
        A_sym = a + a.T
    else:
        A_sym = None

    while time.time() < fw_deadline:
        # Construct linear cost matrix C for the Frank-Wolfe step
        # C[i, k] represents the marginal cost of assigning facility i to location k
        # given the current assignment p.
        # C = A @ B_p.T + A.T @ B_p (general case)
        # If B is symmetric, C = (A + A.T) @ B_p
        
        B_p = b[p]  # Rows of B permuted by p. B_p[i, k] = B[p[i], k]
        
        if is_sym:
            C = A_sym @ B_p
        else:
            C = a @ B_p.T + a.T @ B_p
        
        # Solve linear assignment to get the next vertex of the Birkhoff polytope
        row_ind, col_ind = linear_sum_assignment(C)
        
        # New permutation
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        # Evaluate new cost
        new_cost = calc_cost(p_new)
        
        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improvement from the global linearization step.
            # We might be stuck in a basin where the linear approximation
            # doesn't lead to an immediate improvement, or we've converged.
            # Perform a few random swaps to perturb the solution before
            # attempting another linearization step.
            improved = False
            for _ in range(30):
                i, j = np.random.choice(n, 2, replace=False)
                p_swap = p.copy()
                p_swap[i], p_swap[j] = p_swap[j], p_swap[i]
                swap_cost = calc_cost(p_swap)
                if swap_cost < current_cost:
                    p = p_swap
                    current_cost = swap_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if not improved:
                # If even random swaps don't help immediately, we might be
                # in a very tight local optimum. Break to SA phase which
                # handles this better with temperature.
                break

    # Phase 2: Simulated Annealing (Fine-grained search)
    # Start SA from the best solution found in Phase 1.
    p = best_p.copy()
    current_cost = best_cost
    
    # SA Parameters
    # We have ~70s left.
    # Target ~1-2 million iterations.
    # Let's estimate iterations per second.
    # calc_cost is ~50us. Loop overhead ~10us. Total ~60us/iter.
    # 70s / 60us = ~1.1M iterations.
    
    T = 1000.0
    T_end = 1.0
    alpha = 0.99998  # Very slow cooling to maintain exploration
    
    # Precompute exp for faster acceptance? No, T changes.
    # Use log-exp trick for stability? delta/T can be large.
    
    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        p[i], p[j] = p[j], p[i]
        new_cost = calc_cost(p)
        delta = new_cost - current_cost
        
        if delta < 0:
            accept = True
        else:
            # Accept with probability exp(-delta / T)
            # If delta/T is very large, exp is 0.
            # If delta/T is small, exp is ~1.
            # Use log to avoid underflow?
            # log_prob = -delta / T
            # if log_prob < -700: prob = 0
            # else: prob = exp(log_prob)
            
            # Simple check
            if delta > T * 700:
                accept = False
            else:
                accept = (np.random.rand() < np.exp(-delta / T))
                
        if accept:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            p[i], p[j] = p[j], p[i]
            
        T *= alpha
        if T < T_end:
            T = T_end

    return best_p.tolist()
