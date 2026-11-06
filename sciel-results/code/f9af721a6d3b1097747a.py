import time
import numpy as np
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _compute_covering_radius_and_grads(pts):
    """
    Computes the covering radius and the gradient of the covering radius w.r.t each point.
    Returns:
    radius: The covering radius in radians.
    grads: Array of shape (N, 3) containing the gradient of the radius w.r.t each point.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0 * math.pi / 180.0, np.zeros_like(pts)

    simplices = hull.simplices
    n_faces = len(simplices)
    
    # Precompute circumcenters and radii for all facets
    centers = np.zeros((n_faces, 3))
    min_dots = np.zeros((n_faces, 3))
    radii = np.zeros(n_faces)
    
    for i, simplex in enumerate(simplices):
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Cross product for normal
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            radii[i] = 0.0
            continue
        n = n / n_norm
        
        # Ensure outward normal
        if np.dot(n, v0) < 0:
            n = -n
            
        centers[i] = n
        
        d0 = np.dot(v0, n)
        d1 = np.dot(v1, n)
        d2 = np.dot(v2, n)
        
        min_dots[i, 0] = d0
        min_dots[i, 1] = d1
        min_dots[i, 2] = d2
        
        min_d = np.min(min_dots[i])
        # Radius is arccos(min_d)
        # Clip to avoid domain errors
        min_d = np.clip(min_d, -1.0, 1.0)
        radii[i] = np.arccos(min_d)
        
    max_radius = np.max(radii)
    
    # Identify active facets
    # Use a threshold to include facets close to the max
    # This smooths the gradient
    threshold = max_radius - 0.001 
    active_mask = radii > threshold
    
    if not np.any(active_mask):
        return max_radius, np.zeros_like(pts)
    
    grads = np.zeros_like(pts)
    
    active_indices = np.where(active_mask)[0]
    
    for i in active_indices:
        simplex = simplices[i]
        n = centers[i]
        
        d0, d1, d2 = min_dots[i]
        
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        g0 = n - np.dot(n, v0) * v0
        g1 = n - np.dot(n, v1) * v1
        g2 = n - np.dot(n, v2) * v2
        
        # Normalize gradients to have similar magnitude
        g0_norm = np.linalg.norm(g0)
        g1_norm = np.linalg.norm(g1)
        g2_norm = np.linalg.norm(g2)
        
        if g0_norm > 1e-9: g0 /= g0_norm
        if g1_norm > 1e-9: g1 /= g1_norm
        if g2_norm > 1e-9: g2 /= g2_norm
        
        # Accumulate
        grads[simplex[0]] += g0
        grads[simplex[1]] += g1
        grads[simplex[2]] += g2
        
    return max_radius, grads

def _compute_soft_covering_radius_and_grads(pts, samples, beta):
    """
    Computes a soft approximation of the covering radius using a log-sum-exp (soft-min) of distances.
    This provides a smooth objective and gradient for L-BFGS.
    """
    pts = _normalize(pts)
    
    # Compute angular distances from samples to all points
    # dists[i, j] = arccos(samples[i] . pts[j])
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Soft min of distances for each sample
    # soft_min_i = -1/beta * log(sum_j exp(-beta * dists[i, j]))
    # We want to minimize the max of these soft mins.
    # Let's use the sum of soft mins as the objective.
    
    scaled = -beta * dists
    # Use log-sum-exp trick for numerical stability
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled), axis=1, keepdims=True))
    soft_mins = -log_sum_exp / beta
    
    loss = np.sum(soft_mins)
    
    # Gradient
    # d loss / d dists[i, j] = -1/beta * exp(-beta * dists[i, j]) / sum_k exp(-beta * dists[i, k])
    # = - exp(scaled[i, j]) / sum_k exp(scaled[i, k])
    
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled, axis=1, keepdims=True)
    weights = exp_scaled / sum_exp
    
    # d loss / d dots[i, j] = d loss / d dists[i, j] * d dists[i, j] / d dots[i, j]
    # d arccos(x) / dx = -1 / sqrt(1 - x^2)
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots**2))
    grad_dots = -weights / denom
    
    # d loss / d pts[j, k] = sum_i grad_dots[i, j] * samples[i, k]
    # grad_dots shape (M, N), samples shape (M, 3)
    # result shape (N, 3)
    grad_pts = grad_dots.T @ samples
    
    # Project gradient to tangent space
    # grad_tangent[j, k] = grad_pts[j, k] - dot(grad_pts[j, :], pts[j, :]) * pts[j, k]
    grad_dot_pts = np.sum(grad_pts * pts, axis=1, keepdims=True)
    grad_tangent = grad_pts - grad_dot_pts * pts
    
    return loss, grad_tangent

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initialization
    best_P = None
    best_radius = 180.0 * math.pi / 180.0
    
    rng = np.random.default_rng(42)
    
    # Initializations
    initializations = [
        _fibonacci_sphere(N, seed_noise=0.0, seed=42),
        _fibonacci_sphere(N, seed_noise=0.05, seed=1),
        _fibonacci_sphere(N, seed_noise=0.1, seed=2)
    ]
    
    # Sample points for soft optimization
    # Use a moderate number to balance accuracy and speed
    M_samples = 20000
    samples = _fibonacci_sphere(M_samples)
    
    # Add more initializations for better coverage
    initializations.append(_fibonacci_sphere(N, seed_noise=0.15, seed=3))
    initializations.append(_fibonacci_sphere(N, seed_noise=0.2, seed=4))
    
    for init_P in initializations:
        if time.time() > deadline - 40.0:
            break
            
        P = init_P.copy()
        
        # Phase 1: L-BFGS with soft objective
        # Start with lower beta (smoother), then increase
        betas = [10.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            if time.time() > deadline - 15.0:
                break
                
            # Define objective and gradient
            def obj_grad(P_flat):
                P_local = P_flat.reshape(-1, 3)
                loss, grad = _compute_soft_covering_radius_and_grads(P_local, samples, beta)
                return loss, grad.flatten()
            
            # L-BFGS optimization
            try:
                res = minimize(
                    obj_grad,
                    P.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
                )
                if res.x is not None:
                    P = res.x.reshape(-1, 3)
                    P = _normalize(P)
            except Exception:
                pass
        
        # Phase 2: Riemannian Gradient Descent / Annealing with exact hull gradient
        # This phase uses the exact geometric definition to polish the solution
        step_size = 0.03
        cool_rate = 0.998
        temp = 0.005
        
        for it in range(2000):
            if time.time() > deadline - 5.0:
                break
                
            radius, grads = _compute_covering_radius_and_grads(P)
            
            if radius < best_radius:
                best_radius = radius
                best_P = P.copy()
                
            # Add noise to escape local minima
            noise = rng.normal(0, temp, (N, 3))
            # Project noise to tangent space
            noise = noise - np.sum(noise * P, axis=1, keepdims=True) * P
            
            grads += noise
            
            # Move
            new_P = P + step_size * grads
            new_P = _normalize(new_P)
            
            new_radius, _ = _compute_covering_radius_and_grads(new_P)
            
            # Metropolis acceptance
            delta_r = new_radius - radius
            if delta_r < 0:
                P = new_P
            else:
                if rng.random() < math.exp(-delta_r / max(temp, 1e-10)):
                    P = new_P
            
            # Cool down
            temp *= cool_rate
            step_size *= 0.9995
    
    if best_P is None:
        best_P = _fibonacci_sphere(N)
        
    # Final polish: run a few more L-BFGS iterations with high beta
    if time.time() < deadline - 3.0:
        for beta in [300.0, 500.0]:
            if time.time() > deadline - 2.0:
                break
            try:
                def final_obj_grad(P_flat):
                    P_local = P_flat.reshape(-1, 3)
                    loss, grad = _compute_soft_covering_radius_and_grads(P_local, samples, beta)
                    return loss, grad.flatten()
                
                res = minimize(
                    final_obj_grad,
                    best_P.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                if res.x is not None:
                    P_polished = res.x.reshape(-1, 3)
                    P_polished = _normalize(P_polished)
                    # Check if improved
                    final_radius, _ = _compute_covering_radius_and_grads(P_polished)
                    if final_radius < best_radius:
                        best_radius = final_radius
                        best_P = P_polished
            except Exception:
                pass
    
    return best_P
# EVOLVE-BLOCK-END
