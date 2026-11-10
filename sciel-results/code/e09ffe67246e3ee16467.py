import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    # Initial configuration: 4x4x4 grid, pick 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Randomly select 50
    indices = np.random.choice(len(grid), N, replace=False)
    C = grid[indices].copy()
    
    # Precompute initial distances
    # D[i, j] = distance between i and j
    # W[i, k] = distance from i to wall k
    D = np.zeros((N, N))
    W = np.zeros((N, 3))
    
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(C[i] - C[j])
            D[i, j] = d
            D[j, i] = d
        for k in range(3):
            W[i, k] = 0.5 - abs(C[i, k])
            
    p = 10  # Power for pair repulsion
    q = 10  # Power for wall repulsion
    
    # Initial energy
    E = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if D[i, j] > 1e-9:
                E += 2.0 / (D[i, j] ** p) # Factor 2 for symmetry? No, sum over i<j.
            # Actually, sum_{i<j} 1/d^p.
            # My loop adds 1/d^p for each pair.
            # So E += 1/d^p.
            # Let's re-calculate.
            pass
            
    # Recalculate E properly
    E = 0.0
    for i in range(N):
        for j in range(i+1, N):
            E += 1.0 / (D[i, j] ** p)
        for k in range(3):
            if W[i, k] > 1e-9:
                E += 1.0 / (W[i, k] ** q)
            else:
                E += 1e10 # Penalty for touching wall?
                
    # If any distance is 0, energy is inf
    if np.any(D < 1e-9) or np.any(W < 1e-9):
         E = 1e10

    T = 0.05
    alpha = 0.999
    steps_per_temp = 100
    
    best_E = E
    best_C = C.copy()
    best_r = _calc_radius(C)
    
    iter_count = 0
    while time.time() - start_time < deadline:
        # One temperature step
        for _ in range(steps_per_temp):
            i = np.random.randint(N)
            # Propose move
            # Step size
            step = 0.01 * T + 0.001
            move = np.random.normal(0, step, 3)
            C_new = C[i].copy()
            C_new += move
            
            # Check bounds
            if np.any(C_new < -0.5) or np.any(C_new > 0.5):
                continue # Reject move outside cube
                
            # Calculate delta E
            dE = 0.0
            
            # Pair terms involving i
            for j in range(N):
                if i == j: continue
                d_old = D[i, j]
                d_new = np.linalg.norm(C_new - C[j])
                
                if d_old < 1e-9:
                    term_old = 1e10
                else:
                    term_old = 1.0 / (d_old ** p)
                    
                if d_new < 1e-9:
                    term_new = 1e10
                else:
                    term_new = 1.0 / (d_new ** p)
                    
                dE += term_new - term_old
                
            # Wall terms for i
            for k in range(3):
                w_old = W[i, k]
                w_new = 0.5 - abs(C_new[k])
                
                if w_old < 1e-9:
                    term_old = 1e10
                else:
                    term_old = 1.0 / (w_old ** q)
                    
                if w_new < 1e-9:
                    term_new = 1e10
                else:
                    term_new = 1.0 / (w_new ** q)
                    
                dE += term_new - term_old
                
            # Accept/Reject
            if dE < 0 or np.random.random() < np.exp(-dE / T):
                # Accept
                # Update C
                C[i] = C_new
                
                # Update D
                for j in range(N):
                    if i == j: continue
                    D[i, j] = np.linalg.norm(C_new - C[j])
                    D[j, i] = D[i, j]
                    
                # Update W
                for k in range(3):
                    W[i, k] = 0.5 - abs(C_new[k])
                    
                # Update E
                E += dE
                
                # Update best
                if E < best_E:
                    best_E = E
                    best_C = C.copy()
                    # Calculate radius for best
                    # This is expensive, do it less frequently or at the end?
                    # For now, update best_E. We will calc radius at the end or periodically.
                    # Let's calc radius every 1000 iterations to track progress?
                    # Or just return best_C at the end.
                    
        T *= alpha
        iter_count += steps_per_temp
        
        # Periodic radius check
        if iter_count % 1000 == 0:
             r = _calc_radius(C)
             if r > best_r:
                 best_r = r
                 best_C = C.copy()
                 
    # Final polish?
    # Return best configuration
    return best_C

def _calc_radius(C):
    N = len(C)
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(C[i] - C[j])
            if d < min_dist:
                min_dist = d
        for k in range(3):
            w = 0.5 - abs(C[i, k])
            if w < min_dist:
                min_dist = w
    return min_dist / 2.0
