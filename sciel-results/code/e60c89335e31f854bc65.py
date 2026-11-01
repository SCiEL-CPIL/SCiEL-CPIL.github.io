import numpy as np
import math
import time

N = 150

def energy_and_grad(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist[dist < 1e-12] = 1e-12
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    energy = -np.sum(np.log(dist[mask]))
    
    dist_sq = dist**2
    dist_sq[dist_sq < 1e-24] = 1e-24
    
    term = diff / dist_sq[:, :, np.newaxis]
    grad_sum = np.sum(term, axis=1)
    grad = -grad_sum
    
    return energy, grad

def optimize(points, start_time, deadline, max_iter=5000, eta_init=0.01):
    points = points.copy()
    energy, grad = energy_and_grad(points)
    eta = eta_init
    best_energy = energy
    best_points = points.copy()
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        dot = np.sum(grad * points, axis=1, keepdims=True)
        v = grad - dot * points
        
        new_points = points + eta * v
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_energy, new_grad = energy_and_grad(new_points)
        
        if new_energy < energy:
            points = new_points
            energy = new_energy
            grad = new_grad
            
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
            
            # Adaptive step size
            if i % 10 == 0:
                eta = min(eta * 1.1, 0.05)
        else:
            eta *= 0.5
            if eta < 1e-6:
                break
                
    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points_fib
    best_energy, _ = energy_and_grad(points_fib)
    
    if time.time() < deadline:
        p, e = optimize(points_fib, start_time, deadline, max_iter=3000)
        if e < best_energy:
            best_points = p
            best_energy = e
            
    # Random 1
    rng = np.random.default_rng(42)
    points_rand = rng.random((N, 3))
    points_rand = points_rand / np.linalg.norm(points_rand, axis=1, keepdims=True)
    
    if time.time() < deadline:
        p, e = optimize(points_rand, start_time, deadline, max_iter=3000)
        if e < best_energy:
            best_points = p
            best_energy = e

    # Random 2
    rng = np.random.default_rng(123)
    points_rand2 = rng.random((N, 3))
    points_rand2 = points_rand2 / np.linalg.norm(points_rand2, axis=1, keepdims=True)
    
    if time.time() < deadline:
        p, e = optimize(points_rand2, start_time, deadline, max_iter=3000)
        if e < best_energy:
            best_points = p
            best_energy = e

    # Polish
    if time.time() < deadline:
        p, e = optimize(best_points, start_time, deadline, max_iter=1000, eta_init=1e-4)
        if e < best_energy:
            best_points = p
            best_energy = e

    return best_points
