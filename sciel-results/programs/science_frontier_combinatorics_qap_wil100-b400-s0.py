"""
Simulated Annealing solver for QAPLIB wil100.
Uses multi-start SA with a final local search polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j.
    """
    pi = p[i]
    pj = p[j]
    
    # T1: sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    # We compute sum for all k, then subtract k=i and k=j terms.
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Correction for k=i: (a[i,i] - a[i,j]) * (b[pi, pj] - b[pi, pi])
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    # Correction for k=j: (a[j,i] - a[j,j]) * (b[pj, pj] - b[pj, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # T2: sum_{k != i, j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Correction for k=i: (a[i,i] - a[j,i]) * (b[pj, pi] - b[pi, pi])
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    # Correction for k=j: (a[i,j] - a[j,j]) * (b[pj, pj] - b[pi, pj])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # T3: (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _local_search_polish(a, b, p, current_cost, time_limit):
    """
    Perform First-Improvement local search.
    """
    n = len(p)
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                return p, current_cost
            for j in range(i + 1, n):
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # Restart inner loop? Or just continue? 
                    # Continuing is faster but might miss better moves early in the scan.
                    # Restarting is safer for First-Improvement to ensure we find the *first* improvement
                    # in the next full pass, but usually just updating and continuing is fine for "steepest" like behavior.
                    # However, standard First-Improvement often restarts the scan after a move to ensure 
                    # we don't miss a better immediate move that becomes available due to the swap.
                    # For speed, we just continue, but this is "Greedy" in the sense of taking the first 
                    # improvement found in the current pass.
                    break 
            if improved:
                break
                
    return p, current_cost

def _sa_run(a, b, p_init, time_limit, T_start, T_end):
    """
    Run Simulated Annealing for a given time limit.
    """
    n = len(p_init)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = current_cost + delta
        
        # Adaptive Temperature
        elapsed = time.time() - start_time
        # Avoid division by zero or negative time
        if elapsed == 0:
            T = T_start
        else:
            T = T_start * (T_end / T_start) ** (elapsed / time_limit)
            
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Multi-start configuration
    num_restarts = 5
    time_per_restart = total_deadline / num_restarts
    
    best_p = None
    best_cost = float('inf')
    
    for r in range(num_restarts):
        # Check time
        if time.time() - start_time > total_deadline - 5:
            break
            
        # Generate initial solution
        # Mix of greedy and random to ensure diversity
        if r == 0:
            p_init = _greedy_initial(a, b)
        else:
            # Start with greedy, then apply some random swaps to perturb
            p_init = _greedy_initial(a, b)
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                p_init[i], p_init[j] = p_init[j], p_init[i]
                
        # Run SA
        # Leave some time for polish
        sa_time = time_per_restart * 0.8
        polish_time = time_per_restart * 0.2
        
        p_sa, cost_sa = _sa_run(a, b, p_init, sa_time, T_start=1000.0, T_end=1.0)
        
        # Run Polish
        p_polished, cost_polished = _local_search_polish(a, b, p_sa, cost_sa, polish_time)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
            
    # Final check
    if best_p is None:
        # Fallback
        p_init = _greedy_initial(a, b)
        best_p, best_cost = _local_search_polish(a, b, p_init, _qap_cost(a, b, p_init), 5.0)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
