import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    # Hard deadline: 120s. We want to return well before, say 115s.
    deadline = 115.0
    
    # Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    
    # Optimization Parameters
    k = 15  # Higher k makes repulsion more local/sharp
    eta = 0.005  # Start with smaller step to be stable
    eta_min = 1e-8
    decay = 0.9995
    
    steps = 0
    max_steps = 200000  # Allow many steps
    
    while steps < max_steps:
        if time.time() > deadline:
            break
            
        dot = np.dot(points, points.T)
        np.fill_diagonal(dot, 1.0)
        
        u = 1.0 - dot
        u = np.clip(u, 1e-12, 2.0)
        
        # Gradient of sum_{j!=i} (1 - p_i p_j)^-k wrt p_i
        # = sum_{j!=i} k (1 - p_i p_j)^(-k-1) p_j
        weights = k * (u ** (-k - 1))
        np.fill_diagonal(weights, 0.0)
        
        grads = np.dot(weights, points)
        
        # Update
        new_points = points - eta * grads
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        points = new_points
        
        eta *= decay
        if eta < eta_min:
            eta = eta_min
            
        steps += 1
        
        # If we have time, we could try to verify or just continue
        # The energy minimization is monotonic if step size is small enough.
        
    return points
