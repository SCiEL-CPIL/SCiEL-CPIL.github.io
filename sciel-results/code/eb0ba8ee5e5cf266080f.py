# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115.0
    
    # Precompute sums for greedy init
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    def compute_cost(perm):
        # perm is a numpy array of locations for each facility
        b_perm = b[np.ix_(perm, perm)]
        return np.sum(a * b_perm)

    def greedy_init():
        # Assign facilities in order of decreasing total flow
        # To locations in order of increasing total distance (central)
        heavy = np.argsort(-a_sum, kind="stable")
        central = np.argsort(b_sum, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    best_p = greedy_init()
    best_cost = compute_cost(best_p)
    
    # Simulated Annealing Parameters
    T0 = 1000.0  # Initial temperature
    alpha = 0.995  # Cooling rate
    Tmin = 1e-4
    max_iter_per_run = 20000
    
    # We will run multiple SA runs until time runs out
    while time.time() - start_time < time_budget - 2.0:
        p = greedy_init()
        # Add some random noise to initial solution to diversify
        for _ in range(n // 10):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
            
        current_cost = compute_cost(p)
        T = T0
        
        # Inner SA loop
        while T > Tmin and time.time() - start_time < time_budget - 2.0:
            # Pick random swap
            i, j = random.sample(range(n), 2)
            if i == j:
                continue
                
            # Calculate delta
            li = p[i]
            lj = p[j]
            
            # Delta calculation for swap (i, j)
            # Using vectorized dot products
            S1 = np.dot(a[i, :], b[li, p])
            S2 = np.dot(a[:, i], b[p, li])
            S3 = np.dot(a[j, :], b[lj, p])
            S4 = np.dot(a[:, j], b[p, lj])
            
            N1 = np.dot(a[i, :], b[lj, p])
            N2 = np.dot(a[:, i], b[p, lj])
            N3 = np.dot(a[j, :], b[li, p])
            N4 = np.dot(a[:, j], b[p, li])
            
            delta = (N1 + N2 + N3 + N4) - (S1 + S2 + S3 + S4)
            
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
            T *= alpha
            
            # Check time
            if time.time() - start_time > time_budget - 2.0:
                break

    return best_p.tolist()
# EVOLVE-BLOCK-END
