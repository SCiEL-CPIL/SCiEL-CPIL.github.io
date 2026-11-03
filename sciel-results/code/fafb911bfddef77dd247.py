import numpy as np
import time

def solve(a, b):
    n = len(a)
    # 1. Initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute S = A + A.T
    S = a + a.T
    
    # Helper to get V_i (row of B indexed by p[i])
    # V_i = b[p[i], :]
    
    # Compute initial cost
    # Cost = 0.5 * sum(S[i] dot b[p[i], :])
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Simulated Annealing
    T = 1.0 # Start temp
    T_min = 0.01
    # Cooling schedule
    
    # We need to track current cost to accept/reject
    # current_cost = ...
    
    # SA Loop
    while time.time() < deadline - 5.0:
        # Pick random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j: continue
        
        vi = b[p[i], :]
        vj = b[p[j], :]
        
        # Delta = (S[i] - S[j]) dot (vj - vi)
        delta = np.dot(S[i] - S[j], vj - vi)
        
        # Acceptance
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Swap
            p[i], p[j] = p[j], p[i]
            # Update cost? 
            # We can just recompute or track. Tracking is faster.
            # current_cost += delta
            
        # Cooling
        T *= 0.999
        if T < T_min: T = T_min

    # Final Local Search (Hill Climbing)
    # Use the same delta formula
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 1e10
        best_i, best_j = -1, -1
        
        for i in range(n):
            vi = b[p[i], :]
            for j in range(i+1, n):
                vj = b[p[j], :]
                delta = np.dot(S[i] - S[j], vj - vi)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            improved = True
            
    return p.tolist()
