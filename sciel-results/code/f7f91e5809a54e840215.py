"""
Rewritten program: Riemannian Accelerated Gradient Descent with Multi-Start
for minimal logarithmic energy of 150 points on the unit sphere.

Strategy:
1. Generate diverse initial configurations (Fibonacci, Halton, Greedy FPS, Random).
2. Use Riemannian Gradient Descent with Momentum (Nesterov Acceleration) 
   on the sphere manifold. This uses the exponential map to stay on the sphere 
   and leverages gradient information for efficient descent, avoiding the 
   inefficiency of random-walk Simulated Annealing in high dimensions.
3. Implement a "Shake and Restart" mechanism to escape deep local minima 
   by perturbing the best-found configuration and re-optimizing.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize # Kept for potential fallback or hybrid, but we will implement custom RGD

N = 150
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 5.0  # seconds before the deadline to stop and return best

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) 
    and its Riemannian gradient (projected to tangent space).
    """
    n = pts.shape[0]
    
    # Compute pairwise differences and squared distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] 
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient calculation in ambient space:
    # grad_ambient_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # Note: The force from j to i is (x_i - x_j)/d^2. 
    # Energy is -sum log(d). dE/dd = -1/d. 
    # grad E = sum (x_i - x_j) * (-1/d) * (1/d) * (x_i - x_j)/d ... wait.
    # Let's derive: E = - sum_{i<j} log(||x_i - x_j||).
    # dE/dx_i = - sum_{j != i} (1/||x_i - x_j||) * (1/2) * (x_i - x_j) / (||x_i - x_j||/2) ??
    # d log(r)/dx = (1/r) * (x/r) = x/r^2.
    # So d(-log(r))/dx = -x/r^2.
    # Here r = ||x_i - x_j||. Vector is x_i - x_j.
    # grad_i E = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    #           = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    
    # We computed inv_d2.
    # weighted_sum_pts = sum_j inv_d2[i,j] * pts[j]
    # sum_inv_d2 = sum_j inv_d2[i,j]
    # grad_ambient_i = sum_j inv_d2[i,j] * pts[j] - pts[i] * sum_j inv_d2[i,j]
    
    weighted_sum_pts = inv_d2 @ pts 
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad_ambient = weighted_sum_pts - pts * sum_inv_d2[:, np.newaxis]
    
    # Project gradient to tangent space: g_tangent = g - (g . x) * x
    dot_prod = np.sum(grad_ambient * pts, axis=1)
    grad_tangent = grad_ambient - dot_prod[:, np.newaxis] * pts
    
    # Energy calculation: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad_tangent

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed (no gradient)."""
    n = pts.shape[0]
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _exponential_map(pts, vecs):
    """
    Exponential map on the unit sphere.
    pts: (N, 3)
    vecs: (N, 3) tangent vectors
    Returns new points on the sphere.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    sin_term = np.sin(norms_safe)
    cos_term = np.cos(norms_safe)
    
    # Exp_x(v) = x * cos(||v||) + (v/||v||) * sin(||v||)
    # If v is 0, result is x.
    # Formula: x * cos(||v||) + v * (sin(||v||) / ||v||)
    
    new_pts = pts * cos_term + vecs * (sin_term / norms_safe)
    
    # Normalize to ensure unit length (numerical stability)
    new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
    
    return new_pts

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    points = []
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    for i in range(n):
        z = 1 - (2 * i + 1) / n
        r = math.sqrt(1 - z*z)
        theta = 2 * math.pi * i / phi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append([x, y, z])
    pts = np.array(points)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        # u3 is not strictly needed if we use spherical coords, but let's use 3D Halton for better distribution?
        # Standard spherical mapping:
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _greedy_fps_sphere(n, max_iters=1000, seed=0):
    """
    Greedy Farthest Point Sampling on the sphere.
    Starts with a random point, then repeatedly adds the point on the sphere
    that maximizes the minimum distance to existing points.
    This creates a very uniform initial distribution.
    """
    rng = np.random.default_rng(seed)
    pts = []
    # Start with a random point
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    pts.append(v)
    
    for _ in range(1, n):
        # Sample many candidates
        candidates = rng.normal(size=(max_iters, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Calculate min distance from each candidate to existing points
        # dist_sq[i, j] = ||candidate_i - point_j||^2
        # We want to find candidate i that maximizes min_j dist_sq[i, j]
        
        # Vectorized: (max_iters, 1, 3) - (1, len(pts), 3)
        diff = candidates[:, np.newaxis, :] - np.array(pts)[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        min_dists = np.min(dists_sq, axis=1)
        
        best_idx = np.argmax(min_dists)
        pts.append(candidates[best_idx])
        
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_gd(pts, time_limit=10.0, initial_lr=0.01, momentum=0.9):
    """
    Riemannian Gradient Descent with Momentum (Nesterov-like).
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy, grad = _energy_and_grad(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    velocity = np.zeros_like(current_pts)
    lr = initial_lr
    
    while time.time() - start_time < time_limit:
        # Update velocity
        velocity = momentum * velocity - lr * grad
        
        # Propose new point
        proposed_pts = _exponential_map(current_pts, velocity)
        
        # Compute new energy
        proposed_energy, new_grad = _energy_and_grad(proposed_pts)
        
        # Adaptive Learning Rate
        if proposed_energy < current_energy:
            # Accept
            current_pts = proposed_pts
            current_energy = proposed_energy
            # Increase LR slightly
            lr *= 1.05
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
        else:
            # Reject, reduce LR
            lr *= 0.5
            velocity = 0.5 * velocity # Dampen velocity
            
        # Check convergence
        if np.linalg.norm(grad) < 1e-8:
            break
            
        # Safety check for NaN
        if not np.isfinite(current_energy):
            break

    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    best_pts = None
    best_energy = float('inf')
    
    # 1. Generate Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_greedy_fps_sphere(N, seed=1))
    inits.append(_greedy_fps_sphere(N, seed=2))
    
    # Add a random init
    rng = np.random.default_rng(42)
    rand_pts = rng.normal(size=(N, 3))
    inits.append(_normalize(rand_pts))
    
    # 2. Multi-Start Optimization
    # Allocate time per start
    time_per_start = 10.0
    num_starts = len(inits)
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        t_limit = min(time_per_start, remaining)
        
        if t_limit < 1.0:
            continue
            
        try:
            pts_opt, e_opt = _riemannian_gd(init_pts, time_limit=t_limit)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
            
    # 3. Shake and Restart
    # If we have time left, try to escape local minima
    if best_pts is not None and time.time() < deadline:
        rng_shake = np.random.default_rng(123)
        shake_count = 0
        while time.time() < deadline - 2.0 and shake_count < 5:
            # Create a shake: perturb a random subset of points
            shake_pts = best_pts.copy()
            # Pick 20% of points to shake
            num_shake = N // 5
            indices = rng_shake.choice(N, size=num_shake, replace=False)
            
            # Add random noise
            noise = rng_shake.normal(0, 0.5, size=(num_shake, 3))
            shake_pts[indices] += noise
            shake_pts = _normalize(shake_pts)
            
            # Optimize the shaken configuration
            t_limit = min(5.0, deadline - time.time())
            if t_limit < 1.0:
                break
                
            try:
                pts_opt, e_opt = _riemannian_gd(shake_pts, time_limit=t_limit)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = pts_opt.copy()
            except Exception:
                pass
            shake_count += 1
            
    # 4. Final Polish with L-BFGS-B if time permits
    if best_pts is not None and time.time() < deadline - 5.0:
        # L-BFGS-B can be used for final precision
        x0 = best_pts.flatten()
        
        def fun_and_grad(x):
            p = x.reshape(N, 3)
            p = _normalize(p)
            energy, grad = _energy_and_grad(p)
            return energy, grad.flatten()
            
        try:
            # Use a short maxiter to avoid timeout
            max_iter = 100
            res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
            x_opt = res.x.reshape(N, 3)
            x_opt = _normalize(x_opt)
            e_opt = _log_energy_fast(x_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = x_opt.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
    
    # Final projection and validation
    best_pts = _normalize(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                rng_fix = np.random.default_rng(i+j)
                best_pts[j] += 1e-8 * rng_fix.randn(3)
                best_pts[j] = _normalize(best_pts[j])
    
    return best_pts
