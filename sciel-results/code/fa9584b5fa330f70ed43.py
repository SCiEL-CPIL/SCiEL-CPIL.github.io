import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    def local_search(p, cost):
        improved = True
        # Limit iterations to prevent infinite loop if something goes wrong, 
        # though cost should decrease.
        max_iter = 1000 
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Randomize the order of checks to avoid bias? 
            # Or just sweep. Sweep is deterministic.
            # Let's do a sweep.
            for i in range(n - 1):
                if time.time() > deadline - 5.0: # Keep 5s for final checks
                    return p, cost
                for j in range(i + 1, n):
                    loc_i = p[i]
                    loc_j = p[j]
                    
                    diff_a = a[i] - a[j]
                    diff_b = b[loc_j] - b[loc_i]
                    
                    delta = np.dot(diff_a, diff_b)
                    
                    term_i = diff_a[i] * (b[loc_j, loc_i] - b[loc_i, loc_i])
                    term_j = diff_a[j] * (b[loc_j, loc_j] - b[loc_i, loc_j])
                    
                    delta -= term_i
                    delta -= term_j
                    
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break # Restart sweep? Or continue?
                        # Breaking inner loop and continuing outer is "First Improvement" within a pass.
                        # This is standard.
            # If we improved, we loop again.
        return p, cost

    def simulated_annealing(p, cost, time_limit):
        T_start = 10000.0
        T_min = 1.0
        alpha = 0.9995
        
        T = T_start
        current_p = p.copy()
        current_cost = cost
        best_p = p.copy()
        best_cost = cost
        
        while time.time() < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
                
            loc_i = current_p[i]
            loc_j = current_p[j]
            
            diff_a = a[i] - a[j]
            diff_b = b[loc_j] - b[loc_i]
            
            delta = np.dot(diff_a, diff_b)
            
            term_i = diff_a[i] * (b[loc_j, loc_i] - b[loc_i, loc_i])
            term_j = diff_a[j] * (b[loc_j, loc_j] - b[loc_i, loc_j])
            
            delta -= term_i
            delta -= term_j
            
            if delta < 0:
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
            else:
                if T > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_p[i], current_p[j] = current_p[j], current_p[i]
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_p, best_cost

    # 1. Initial Solution
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy = np.argsort(-a_sums, kind='stable')
    central = np.argsort(b_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # 2. Local Search
    p, cost = local_search(p, compute_cost(p))
    best_p = p
    best_cost = cost
    
    # 3. SA Loop with Restarts
    # We will run SA until time is up.
    # To implement restarts, we can just run SA, and if it finishes early (T too low) and time remains, 
    # we perturb and restart.
    
    while time.time() < deadline - 10.0: # Keep 10s for final polish
        # Run SA for a chunk of time or until T drops?
        # Better: Run SA until time limit, but allow restarts.
        # Let's run SA for a fixed duration, say 20s, or until T is low, then restart.
        
        sa_end_time = time.time() + 20.0
        if sa_end_time > deadline - 10.0:
            sa_end_time = deadline - 10.0
            
        p_sa, cost_sa = simulated_annealing(best_p, best_cost, sa_end_time)
        
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa
            
        # Perturb for next restart
        if time.time() < deadline - 10.0:
            # Random 10-20 swaps
            for _ in range(15):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    
    # 4. Final Local Search
    if time.time() < deadline - 2.0:
        p_final, cost_final = local_search(best_p.copy(), best_cost)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
            
    return best_p.tolist()
