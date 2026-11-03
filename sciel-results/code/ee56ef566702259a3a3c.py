import numpy as np
import time
import math

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s or 150s kill
    
    # 1. Initial Solution: Rank Matching
    # Sort facilities by sum of flows (descending)
    # Sort locations by sum of distances (ascending)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy_facilities = np.argsort(-a_sums, kind='stable')
    central_locations = np.argsort(b_sums, kind='stable')
    
    # p[i] is the location for facility i
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    # Calculate initial cost
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Using np.einsum for efficiency
    cost = np.einsum('ij,ij->', a, b[p[:, None], p])
    best_cost = cost
    best_p = p.copy()
    
    # Precompute for delta calculation
    # We need row sums of a? No, we need specific elements.
    # Delta for swapping i and j:
    # loc_i = p[i], loc_j = p[j]
    # delta = sum_{k != i, j} (a[i,k] + a[j,k]) * (b[loc_j, p[k]] - b[loc_i, p[k]])
    
    # To make this fast, we can precompute nothing dynamic, but use vectorized ops.
    # Let's try to optimize the inner loop.
    
    # Temperature schedule
    # Estimate typical move cost
    # Randomly sample some moves to estimate scale
    sample_moves = []
    for _ in range(100):
        i, j = np.random.choice(n, 2, replace=False)
        li, lj = p[i], p[j]
        # Compute delta
        # D = b[lj, p] - b[li, p]
        # W = a[i, :] + a[j, :]
        # dot = np.dot(W, D)
        # Subtract terms for k=i and k=j
        # term_i = W[i] * D[i]
        # term_j = W[j] * D[j]
        # delta = dot - term_i - term_j
        D = b[lj, p] - b[li, p]
        W = a[i, :] + a[j, :]
        delta = np.dot(W, D) - W[i]*D[i] - W[j]*D[j]
        sample_moves.append(abs(delta))
    
    avg_move_cost = np.mean(sample_moves) if sample_moves else 1000
    if avg_move_cost == 0:
        avg_move_cost = 1.0
        
    T = avg_move_cost * 2.0
    alpha = 0.9995 # Cooling rate
    
    # Simulated Annealing Loop
    no_improve_count = 0
    max_no_improve = 50000 # Restart or stop if stuck
    
    while time.time() < deadline:
        i, j = np.random.choice(n, 2, replace=False)
        
        li = p[i]
        lj = p[j]
        
        # Calculate Delta
        # D_k = b[lj, p[k]] - b[li, p[k]]
        # W_k = a[i, k] + a[j, k]
        D = b[lj, p] - b[li, p]
        W = a[i, :] + a[j, :]
        
        # Full dot product
        total_dot = np.dot(W, D)
        
        # Subtract contributions from k=i and k=j
        # For k=i: W[i] * D[i] = (a[i,i] + a[j,i]) * (b[lj, li] - b[li, li])
        # For k=j: W[j] * D[j] = (a[i,j] + a[j,j]) * (b[lj, lj] - b[li, lj])
        term_i = W[i] * D[i]
        term_j = W[j] * D[j]
        
        delta = total_dot - term_i - term_j
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            # Accept move
            cost += delta
            p[i], p[j] = lj, li
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        else:
            no_improve_count += 1
            
        # Cool down
        T *= alpha
        
        # If T becomes very small, we might get stuck.
        # If we have no improvement for a long time, we could restart from best_p
        # But for now, just let it run. If T is too low, exp(-delta/T) is 0 for positive delta.
        if T < 1e-10:
             # Restart with higher T from best solution
             p = best_p.copy()
             cost = best_cost
             T = avg_move_cost * 2.0
             no_improve_count = 0

    return best_p.tolist()
