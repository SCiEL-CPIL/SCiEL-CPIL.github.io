import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Sample points for objective function
    # Using a denser Fibonacci spiral for samples
    M = 20000
    j = np.arange(M) + 0.5
    z_s = 1.0 - 2.0 * j / M
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_s = np.sqrt(1.0 - z_s * z_s)
    Y = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
    
    # Pre-compute Y norms (should be 1)
    # Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    
    # 3. Optimization Loop
    # Objective: Sum of 4th power of distances to nearest point
    # This penalizes large distances more than Lloyd's (2nd power)
    
    eta = 0.05  # Initial learning rate
    best_P = P.copy()
    best_score = np.inf
    
    for t in range(200):
        if time.time() > deadline:
            break
            
        # Compute distances from Y to P
        # D2[k, i] = ||Y[k] - P[i]||^2
        # ||Y - P||^2 = ||Y||^2 + ||P||^2 - 2 Y.P = 1 + 1 - 2 Y.P = 2 - 2 Y.P
        # So we just need Y @ P.T
        YP = Y @ P.T
        D2 = 2.0 - 2.0 * YP
        
        # Find nearest neighbor index for each Y[k]
        I = np.argmin(D2, axis=1)
        
        # Compute gradient for each P[i]
        # J = sum_k D2[k, I[k]]^2  (using D2 as proxy for distance squared, so D2^2 is 4th power of dist)
        # d(J)/d(P[i]) = sum_{k: I[k]==i} 2 * D2[k, i] * d(D2[k, i])/d(P[i])
        # d(D2)/d(P[i]) = -2 * Y[k]
        # So term is: 2 * D2[k, i] * (-2 * Y[k]) = -4 * D2[k, i] * Y[k]
        # We want to minimize J, so we move P in direction -grad
        # grad_Pi = -4 * sum_{k in i} D2[k, i] * Y[k]
        # Update: P_i = P_i - eta * grad_Pi = P_i + 4 * eta * sum_{k in i} D2[k, i] * Y[k]
        
        # Accumulate weighted sums
        # weights[k] = D2[k, I[k]]
        weights = D2[np.arange(M), I]
        
        # Sum Y[k] * weights[k] for each i
        # Use bincount for efficiency? No, Y is 3D.
        # scatter_add
        grad_sum = np.zeros((N, 3))
        np.add.at(grad_sum, I, Y * weights[:, np.newaxis])
        
        # Update P
        P = P + 4.0 * eta * grad_sum
        
        # Normalize P
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / norms
        
        # Decay learning rate
        eta *= 0.99
        
        # Evaluate score (approximate covering radius)
        # Max distance to nearest point
        D2_min = np.min(D2, axis=1)
        # Distance = sqrt(D2_min) ... but D2 is 2-2cos(theta) = 4sin^2(theta/2)
        # theta = 2 * arcsin(sqrt(D2_min)/2)
        # For small theta, theta ~ sqrt(D2_min)
        # Let's compute max angular distance
        # D2_min is 2 - 2cos(theta). cos(theta) = 1 - D2_min/2.
        # theta = arccos(1 - D2_min/2)
        # We want max theta.
        # Note: D2_min is small.
        
        # To avoid arccos instability, use 2*arcsin(sqrt(D2_min)/2)
        dists = 2.0 * np.arcsin(np.sqrt(D2_min) / 2.0)
        current_score = np.max(dists)
        
        if current_score < best_score:
            best_score = current_score
            best_P = P.copy()
            
    return best_P
