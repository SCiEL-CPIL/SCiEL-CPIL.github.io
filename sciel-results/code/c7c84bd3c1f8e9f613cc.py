import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_P = None
    best_score = float('inf')
    
    # Generate sample points once
    np.random.seed(12345)
    u = np.random.rand(2000, 2)
    v = np.random.rand(2000)
    theta = 2 * np.pi * u[:, 0]
    phi = np.arccos(1 - 2 * v)
    samples = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    
    def score_config(P):
        D_s = np.linalg.norm(samples[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
        min_dist = np.min(D_s, axis=1)
        max_dist = np.max(min_dist)
        cos_theta = 1 - 0.5 * max_dist**2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)
    
    # Multiple restarts
    num_restarts = 5
    # We don't strictly need to divide time, the inner loop checks time.
    # But limiting iterations per restart prevents one bad restart from eating all time.
    max_iters_per_restart = 20000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        rad = np.sqrt(1.0 - z * z)
        P = np.column_stack([rad * np.cos(phi), rad * np.sin(phi), z])
        
        # Add noise
        np.random.seed(42 + r)
        P += np.random.normal(0, 0.05, P.shape)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        
        # Optimize
        step_size = 0.05
        decay = 0.9995
        
        for it in range(max_iters_per_restart):
            if time.time() > deadline:
                break
                
            # Calculate forces
            D = P[:, np.newaxis, :] - P[np.newaxis, :, :]
            dist = np.linalg.norm(D, axis=2)
            # Protect against zero distance
            dist = np.maximum(dist, 1e-12)
            np.fill_diagonal(dist, 1.0) # Diagonal is 0, set to 1 so 1/1^3 = 1, but we zero out diagonal in force calc
            inv_dist_cubed = 1.0 / (dist ** 3)
            np.fill_diagonal(inv_dist_cubed, 0.0)
            
            F = np.einsum('ijk,ij->ik', D, inv_dist_cubed)
            
            P += step_size * F
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            
            step_size *= decay
            
            if it % 50 == 0:
                s = score_config(P)
                if s < best_score:
                    best_score = s
                    best_P = P.copy()
                    
    if best_P is None:
        # Fallback to seed
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        rad = np.sqrt(1.0 - z * z)
        best_P = np.column_stack([rad * np.cos(phi), rad * np.sin(phi), z])
        
    return best_P
