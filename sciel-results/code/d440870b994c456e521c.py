import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad(points):
    """
    Calculate logarithmic energy and its gradient.
    points: (N, 3) array
    Returns: energy (float), grad (N, 3) array
    """
    # Compute pairwise distances
    # d[i, j] = ||points[i] - points[j]||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero (though points should be distinct)
    # Mask diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    # We compute sum over all i != j and divide by 2
    # log(1/d) = -log(d)
    # Use np.log(dist)
    
    # For gradient:
    # grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
    # Note: diff[i,j] = x_i - x_j
    # So (x_j - x_i) = -diff[i,j]
    # grad_i = sum_{j != i} -diff[i,j] / dist_sq[i,j]
    
    # Create a mask for i != j
    mask = ~np.eye(N, dtype=bool)
    
    # Gradient calculation
    # term[i,j] = -diff[i,j] / dist_sq[i,j]
    # We need to sum over j for each i.
    # diff is (N, N, 3). dist_sq is (N, N).
    # We want to divide diff by dist_sq[:, :, np.newaxis]
    
    # Avoid division by zero on diagonal
    safe_dist_sq = dist_sq.copy()
    safe_dist_sq[~mask] = 1.0
    
    grad_terms = -diff / safe_dist_sq[:, :, np.newaxis]
    
    # Sum over j (axis 1)
    grad = np.sum(grad_terms, axis=1)
    
    # Energy calculation
    # We need sum of log(1/d) for i < j
    # This is 0.5 * sum_{i != j} log(1/d_ij)
    # = -0.5 * sum_{i != j} log(d_ij)
    
    # Use fsum for accuracy? The prompt says the *evaluator* uses fsum.
    # For optimization, numpy sum is sufficient.
    # But to be safe and consistent, we can compute the scalar energy using a loop or fsum if N is small enough.
    # N=100, so 4950 pairs. A loop is fast enough.
    
    # However, calling math.fsum inside a tight loop might be slow if done every iteration.
    # Let's use numpy sum for the optimization loop, and maybe fsum for the final result check?
    # The score is the raw value.
    
    # Let's stick to numpy for speed in the optimizer.
    log_d = np.log(dist)
    # Sum over upper triangle
    # energy = -np.sum(log_d[mask]) / 2.0
    
    # To match fsum more closely, we can do:
    # idx_i, idx_j = np.triu_indices(N, 1)
    # d_pairs = dist[idx_i, idx_j]
    # energy = -math.fsum(np.log(d_pairs))
    
    # Let's use the fsum version for accuracy, as N=100 is small.
    idx_i, idx_j = np.triu_indices(N, 1)
    d_pairs = dist[idx_i, idx_j]
    # Ensure no zeros
    d_pairs = np.maximum(d_pairs, 1e-15)
    energy = -math.fsum(np.log(d_pairs))
    
    return energy, grad

def _penalty_fun_and_grad(points, lambda_val):
    """
    Add penalty for not being on the sphere.
    """
    energy, grad = _energy_and_grad(points)
    
    norms_sq = np.sum(points * points, axis=1)
    penalty = lambda_val * np.sum((norms_sq - 1.0) ** 2)
    
    # Gradient of penalty:
    # d/dx (lambda * (||x||^2 - 1)^2) = lambda * 2(||x||^2 - 1) * 2x = 4 * lambda * (||x||^2 - 1) * x
    penalty_grad = 4.0 * lambda_val * (norms_sq - 1.0)[:, np.newaxis] * points
    
    total_energy = energy + penalty
    total_grad = grad + penalty_grad
    
    return total_energy, total_grad

def _optimize_single(start_points, lambda_val=100.0, max_iter=500):
    """
    Run L-BFGS-B optimization from a starting point.
    """
    x0 = start_points.flatten()
    
    def fun(x):
        pts = x.reshape(N, 3)
        val, _ = _penalty_fun_and_grad(pts, lambda_val)
        return val
        
    def jac(x):
        pts = x.reshape(N, 3)
        _, g = _penalty_fun_and_grad(pts, lambda_val)
        return g.flatten()
        
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=jac, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        return res.x.reshape(N, 3)
    except Exception:
        return start_points

def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def _polish(points, steps=50, alpha=0.01):
    """
    Final high-precision polish using projected gradient descent.
    """
    pts = points.copy()
    for _ in range(steps):
        _, grad = _energy_and_grad(pts)
        # Project gradient to tangent plane
        norms_sq = np.sum(pts * pts, axis=1)
        # grad_proj = grad - (grad . x) x
        # grad . x = sum(grad * pts, axis=1)
        dot = np.sum(grad * pts, axis=1)
        grad_proj = grad - dot[:, np.newaxis] * pts
        
        # Update
        pts = pts - alpha * grad_proj
        pts = _project_to_sphere(pts)
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initializations
    inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib)
    
    # 2. Random Restarts
    rng = np.random.default_rng(42)
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        # Generate random points on sphere
        pts = rng.normal(size=(N, 3))
        pts = _project_to_sphere(pts)
        inits.append(pts)
        
    # 3. Geodesic-like (Icosahedron vertices replicated + noise)
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Repeat and perturb to get 100 points
    # 12 * 8 = 96, add 4 more?
    # Let's just use the 12 vertices and interpolate?
    # Simpler: Just use random for the rest.
    
    best_energy = float('inf')
    best_points = inits[0]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Optimize
        opt_points = _optimize_single(init)
        
        # Polish
        polished_points = _polish(opt_points, steps=20)
        
        # Evaluate
        energy, _ = _energy_and_grad(polished_points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = polished_points
            
    # Final check and return
    # Ensure valid shape and finite values
    if not np.all(np.isfinite(best_points)):
        return inits[0]
        
    return best_points
# EVOLVE-BLOCK-END
