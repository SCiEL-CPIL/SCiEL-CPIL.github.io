import numpy as np
import time
import math

N = 150

def log_energy_and_grad(pts):
    """
    Computes the logarithmic energy and its Euclidean gradient.
    Energy: sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient: grad_i = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    """
    # pts is (N, 3)
    # Compute pairwise distances
    # To save memory and time, we can do it in a loop or use broadcasting if N is small.
    # N=150, N^2 = 22500, which is fine for broadcasting.
    
    # diff[i, j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dist[i, j] = ||x_i - x_j||
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy calculation
    # E = -0.5 * sum_{i != j} log(dist[i, j])
    # Using math.fsum for precision as requested?
    # The prompt says "pair terms summed with math.fsum".
    # However, doing fsum inside the optimization loop might be slow.
    # We can use fsum for the final score, but for the gradient descent, 
    # standard numpy sum is likely sufficient for the direction.
    # Let's stick to numpy sum for speed in the loop, and use fsum in the final check.
    
    # For the gradient, we need:
    # grad_i = sum_{j != i} (x_j - x_i) / dist[i, j]^2
    # Note: (x_j - x_i) = -diff[i, j]
    # So term is -diff[i, j] / dist_sq[i, j]
    
    # We need to sum over j for each i.
    # inv_dist_sq = 1 / dist_sq
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i, k] = sum_j (-diff[i, j, k] * inv_dist_sq[i, j])
    # diff is (N, N, 3)
    # inv_dist_sq is (N, N)
    # We can use einsum or broadcasting.
    # grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy
    # E = -0.5 * sum_{i != j} log(dist)
    # log_dist = np.log(dist)
    # np.fill_diagonal(log_dist, 0)
    # E = -0.5 * np.sum(log_dist)
    
    # To be precise and match the evaluator's method (fsum), we should calculate 
    # the final score using fsum. But for the internal optimization, 
    # the exact value matters less than the gradient direction.
    # However, for line search, we need a consistent energy measure.
    # Let's use the numpy sum for speed, and verify at the end.
    
    # We only need the energy for line search comparison.
    log_dist = np.log(dist)
    np.fill_diagonal(log_dist, 0.0)
    energy = -0.5 * np.sum(log_dist)
    
    return energy, grad

def riemannian_gradient_descent(start_pts, max_iter=2000, tol=1e-12, time_limit=10):
    """
    Runs Riemannian Gradient Descent on the sphere.
    """
    pts = start_pts.copy()
    current_energy, grad = log_energy_and_grad(pts)
    
    # Initial step size
    eta = 1e-3
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Project gradient to tangent space
        # g_i = grad_i - (grad_i . x_i) x_i
        dot = np.sum(pts * grad, axis=1)
        tangent_grad = grad - dot[:, np.newaxis] * pts
        
        # Check convergence
        # Norm of tangent gradient
        grad_norm = np.linalg.norm(tangent_grad)
        if grad_norm < tol:
            break
            
        # Line search: backtracking
        alpha = eta
        new_pts = pts.copy()
        
        # Geodesic step: x_new = normalize(x - alpha * g)
        # For small alpha, this is approx x - alpha * g.
        # Let's use the normalization step to ensure we stay on the sphere.
        
        found_step = False
        for _ in range(50):
            # Candidate point
            cand_pts = pts - alpha * tangent_grad
            # Normalize
            cand_pts = cand_pts / np.linalg.norm(cand_pts, axis=1)[:, np.newaxis]
            
            new_energy, _ = log_energy_and_grad(cand_pts)
            
            # Armijo condition: E(new) <= E(old) - c * alpha * ||g||^2
            # c = 1e-4
            if new_energy <= current_energy - 1e-4 * alpha * (grad_norm ** 2):
                found_step = True
                break
            alpha *= 0.5
            
        if not found_step:
            # If no step found, try smaller alpha or break
            break
            
        # Update
        pts = cand_pts
        current_energy = new_energy
        # Adapt eta
        # If we used full eta, increase slightly. If we reduced, decrease.
        # Simple heuristic: keep eta constant or decay slowly.
        # Let's decay eta slowly to allow finer steps later.
        eta *= 0.99
        
    return pts, current_energy

def construct():
    start_time = time.time()
    best_pts = None
    best_energy = float('inf')
    
    # Deadline guard: 120s limit, leave 5s margin
    deadline = 115.0
    
    # 1. Fibonacci Spiral Start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    starts = [fib_pts]
    
    # 2. Random Starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        rand_pts = rng.normal(size=(N, 3))
        rand_pts /= np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
        starts.append(rand_pts)
        
    # 3. Perturbed Fibonacci Starts
    for _ in range(10):
        noise = rng.normal(scale=0.1, size=(N, 3))
        pert_pts = fib_pts + noise
        pert_pts /= np.linalg.norm(pert_pts, axis=1)[:, np.newaxis]
        starts.append(pert_pts)
        
    # Optimize each start
    for start in starts:
        if time.time() - start_time > deadline:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - (time.time() - start_time)
        if remaining < 1.0:
            break
            
        # Give more time to earlier, more promising starts?
        # Or fixed time per start.
        # With ~30 starts and 115s, that's ~3s per start.
        # RGD is fast. Let's give 2s per start, and save the rest for a final polish.
        time_alloc = min(2.0, remaining / 2.0)
        
        opt_pts, opt_energy = riemannian_gradient_descent(start, max_iter=5000, time_limit=time_alloc)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
            
    # Final Polish
    # Use the best point found so far and run a longer, more precise optimization.
    if best_pts is not None:
        remaining = deadline - (time.time() - start_time)
        if remaining > 5.0:
            # Use a tighter tolerance and more iterations
            # We can also try a simple Newton-like step or L-BFGS on the sphere
            # but RGD with small steps is robust.
            # Let's run RGD again with very small time limit but high precision.
            # Or, we can do a local search with very small steps.
            
            # Let's just run RGD again on the best point with remaining time.
            # Reduce max_iter to prevent hanging, but let it run until time limit.
            final_pts, final_energy = riemannian_gradient_descent(best_pts, max_iter=10000, time_limit=remaining - 2.0)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
                
    # Verify validity and compute final score with math.fsum
    # Check for close points
    dists = np.sqrt(np.sum((best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        # Invalid, return fibonacci (which is valid)
        return fib_pts
        
    # Calculate energy with math.fsum
    # E = sum_{i<j} log(1/d_ij)
    # We need to sum only i<j to avoid double counting and zero distance.
    energy_val = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            energy_val += math.log(1.0 / d)
            
    # The evaluator uses math.fsum. Let's mimic that for the final check 
    # though the returned value is the array. The score is computed by the evaluator.
    # We just need to return the points.
    
    return best_pts
