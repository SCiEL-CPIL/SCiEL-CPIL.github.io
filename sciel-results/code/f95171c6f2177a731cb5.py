import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # 1. Initialization: Fibonacci Spiral + Small Random Perturbation
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small noise to break symmetry
    noise = np.random.normal(0, 0.01, P.shape)
    P = P + noise
    P = P / np.linalg.norm(P, axis=1, keepdims=True)

    # 2. Repulsive Optimization
    power = 4.0
    eta = 0.01
    decay = 0.999
    max_iters = 100000
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        weights = power * (dist_sq ** (-power))
        np.fill_diagonal(weights, 0.0)
        
        grad_P = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        P = P - eta * grad_P
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        
        eta *= decay
        
        if np.max(np.abs(grad_P)) < 1e-10:
            break
            
    # 3. Polish Phase
    power = 8.0
    eta = 0.002
    for it in range(20000):
        if time.time() - start_time > time_limit:
            break
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        weights = power * (dist_sq ** (-power))
        np.fill_diagonal(weights, 0.0)
        grad_P = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        P = P - eta * grad_P
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        eta *= 0.9995

    return P
