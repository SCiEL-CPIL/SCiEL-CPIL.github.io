import numpy as np
import time
import math

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second buffer
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper to calculate energy of a specific point k
    def energy_of_point(k, points):
        # Sum of log(1/distance) for point k with all others
        # Using math.fsum for precision as requested in contract
        diffs = points[k] - points
        dists = np.linalg.norm(diffs, axis=1)
        # Avoid log(0) if points coincide, though unlikely in float
        dists[k] = 1.0 
        # We want sum_{j != k} log(1/d_j) = - sum_{j != k} log(d_j)
        # But the total energy is sum_{i<j}. 
        # If we define E_total = 0.5 * sum_{i != j} log(1/|x_i - x_j|), 
        # then E_k = sum_{j != k} log(1/|x_k - x_j|) is exactly the contribution of point k to the double sum.
        # The total energy is 0.5 * sum_k E_k.
        # For SA, we only need the change in total energy, which is 0.5 * Delta_E_k.
        # So we can just minimize sum_{j != k} log(1/|x_k - x_j|).
        
        # Filter out zero distances just in case
        valid = dists > 1e-9
        if not np.all(valid):
            return float('inf')
        
        # log(1/d) = -log(d)
        # Use fsum for the sum
        terms = -np.log(dists[valid])
        return math.fsum(terms)

    # Calculate initial total energy
    # E_total = 0.5 * sum_k E_k
    def total_energy(points):
        e_sum = 0.0
        for k in range(N):
            e_sum += energy_of_point(k, points)
        return 0.5 * e_sum

    current_energy = total_energy(x)
    best_x = x.copy()
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    # Number of steps: We have ~110 seconds.
    # One step involves O(N) energy diff calculation. N=100.
    # Let's estimate: 100 ops is very fast. We can do millions of steps.
    # Let's set a max steps limit and a time limit.
    max_steps = 5000000
    
    steps = 0
    rng = np.random.default_rng(42)
    
    while steps < max_steps and time.time() < deadline - 10.0:
        # Temperature schedule: linear or geometric? Geometric is common.
        # Let's use a simple linear cooling for simplicity, or geometric.
        # progress = steps / max_steps
        # T = T_start * (T_end / T_start) ** progress
        # But we don't know when we will stop exactly due to time.
        # Let's just use a fixed number of steps for SA, then polish.
        
        # Let's run SA for a fixed number of steps, say 200,000
        if steps >= 200000:
            break
            
        progress = steps / 200000.0
        T = T_start * (T_end / T_start) ** progress
        
        # Pick random point
        k = rng.integers(0, N)
        
        # Generate random tangent direction
        # Random vector v
        v = rng.normal(size=3)
        # Project to tangent plane at x[k]
        v = v - np.dot(v, x[k]) * x[k]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        v = v / v_norm
        
        # Step size: scale with temperature?
        # Typical step size on unit sphere
        step_size = 0.1 * (T / T_start + 0.1) 
        # Or fixed step size? Fixed is often better for SA if T controls acceptance.
        step_size = 0.05 
        
        new_xk = x[k] + step_size * v
        new_xk = new_xk / np.linalg.norm(new_xk)
        
        # Calculate Delta E
        # Old E_k
        old_E_k = energy_of_point(k, x)
        
        # Temporarily update x
        old_xk = x[k]
        x[k] = new_xk
        new_E_k = energy_of_point(k, x)
        
        delta_E = 0.5 * (new_E_k - old_E_k)
        
        # Acceptance
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_energy += delta_E
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
        else:
            # Reject, revert
            x[k] = old_xk
            
        steps += 1
        
    # 2. Local Polish: Gradient Descent
    # We have ~10 seconds left.
    # Perform gradient descent steps.
    # Calculate gradients for all points.
    
    def calculate_gradients(points):
        grads = np.zeros_like(points)
        for i in range(N):
            # Grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
            # Vectorized:
            diffs = points[i] - points # shape (N, 3)
            dists_sq = np.sum(diffs**2, axis=1)
            dists_sq[i] = 1.0 # Avoid div by zero
            # We want to sum (x_i - x_j) / dist_sq
            # This is equivalent to sum (x_i / dist_sq - x_j / dist_sq)
            # But simpler to just do the sum.
            # Note: diffs[i] is 0.
            inv_dists_sq = 1.0 / dists_sq
            # Set i-th term to 0 to avoid self-interaction in the sum formula if not careful
            # Actually, the formula sum_{j != i} works.
            # grads[i] = sum_{j != i} (points[i] - points[j]) / dists_sq[j]
            # Let's compute the sum over all j, the i-th term is 0/1 = 0.
            weighted_diffs = diffs * inv_dists_sq[:, np.newaxis]
            grads[i] = np.sum(weighted_diffs, axis=0)
            
            # Project to tangent plane
            # g_tan = g - (g . x_i) x_i
            dot = np.dot(grads[i], points[i])
            grads[i] = grads[i] - dot * points[i]
            
        return grads

    # Gradient Descent Loop
    lr = 0.01
    # We will do a few steps with decreasing learning rate
    for _ in range(500):
        if time.time() > deadline:
            break
            
        grads = calculate_gradients(best_x)
        # Update
        new_x = best_x - lr * grads
        # Normalize
        norms = np.linalg.norm(new_x, axis=1)
        new_x = new_x / norms[:, np.newaxis]
        
        # Calculate new energy
        new_energy = total_energy(new_x)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_x = new_x
            # Decrease learning rate if we made progress?
            # Or just keep it constant and let it converge.
            # Let's decrease it slowly.
            lr *= 0.99
        else:
            # If no progress, reduce learning rate more aggressively
            lr *= 0.9
            
    return best_x
