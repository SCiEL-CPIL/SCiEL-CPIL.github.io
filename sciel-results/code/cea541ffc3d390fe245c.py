"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets (circumcenter method)."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            pts = points[simplex]
            a, b, c = pts[0], pts[1], pts[2]
            # Find circumcenter of spherical triangle
            # x · a = x · b = x · c (equidistant)
            # Solve: (a-b)·x = 0, (a-c)·x = 0, |x|=1
            # x is perpendicular to (a-b) and (a-c), so x ∝ (a-b) × (a-c)
            n = np.cross(a - b, a - c)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            x = n / n_norm
            # Ensure x is on the same side as the triangle
            centroid = (a + b + c) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            # Angular distance from circumcenter to any vertex
            cos_dist = np.clip(np.dot(x, a), -1.0, 1.0)
            ang = np.arccos(cos_dist)
            max_r = max(max_r, ang)
        return max_r
    except Exception:
        return np.pi

def _soft_max_surrogate(points, beta=80.0):
    """
    Fast O(N^2) surrogate for covering radius optimization.
    Minimizes the soft-max of closest-neighbor dot products (maximizes min separation).
    """
    n = len(points)
    dots = points @ points.T
    np.fill_diagonal(dots, -1.0)
    
    # For each point i, compute soft-max of dots[i,j] over j≠i
    max_dots = np.max(dots, axis=1)
    shifted = dots - max_dots[:, np.newaxis]
    exp_shifted = np.exp(beta * shifted)
    exp_shifted = np.clip(exp_shifted, 0, 1e100)
    sum_exp = np.sum(exp_shifted, axis=1)
    sum_exp = np.where(sum_exp < 1e-300, 1e-300, sum_exp)
    log_sum_exp = np.log(sum_exp) + max_dots
    
    # Objective: minimize mean of soft-max values (higher dot = closer = worse)
    objective = np.mean(log_sum_exp)
    
    # Weights for gradient
    weights = exp_shifted / sum_exp[:, np.newaxis]
    
    # Gradient of objective w.r.t. points
    # d objective / d dots[i,j] = beta * weights[i,j]
    # d dots[i,j] / d x_i = x_j, d dots[i,j] / d x_j = x_i
    term1 = beta * (weights @ points)
    col_sums = np.sum(weights, axis=0)
    term2 = beta * col_sums[:, np.newaxis] * points
    gradient = (term1 + term2) / n
    
    return objective, gradient

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _fibonacci_sphere(n, rng=None):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_soft_max(points, max_iters=400, lr_init=0.03, beta=80.0, tol=1e-12):
    """
    Projected gradient ascent to maximize the minimum pairwise dot product
    using a soft-max surrogate.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_obj = _soft_max_surrogate(pts, beta)[0]
    
    lr = lr_init
    momentum = np.zeros_like(pts)
    mu = 0.85
    
    for it in range(max_iters):
        obj, grad = _soft_max_surrogate(pts, beta)
        
        # Project gradient to tangent space
        tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Update momentum
        momentum = mu * momentum - lr * tangent_grad
        
        # Take step
        new_pts = pts + momentum
        new_pts = _project_to_sphere(new_pts)
        
        # Check if objective improved
        new_obj = _soft_max_surrogate(new_pts, beta)[0]
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            lr *= 0.7
            momentum = mu * momentum
        
        # Decay learning rate
        if it % 30 == 0 and lr > 0.0005:
            lr *= 0.92
        
        # Convergence check
        if it > 100 and abs(obj - best_obj) < tol:
            break
    
    return best_pts, best_obj

def _repulsion_energy(points):
    """
    Compute repulsion-based energy: sum of 1/d^2 for nearest neighbors.
    Lower energy = better spread.
    """
    n = len(points)
    dots = points @ points.T
    np.fill_diagonal(dots, -1.0)
    
    # Find nearest neighbor dot product for each point
    max_dots = np.max(dots, axis=1)
    
    # Energy: sum of 1/(1 - max_dot) which is related to 1/d^2
    # For small angles, d ≈ sqrt(2*(1-cos)), so 1/d^2 ≈ 1/(2*(1-cos))
    energy = np.sum(1.0 / np.maximum(1e-10, 1.0 - max_dots))
    
    return energy

def _optimize_repulsion(points, max_iters=300, lr_init=0.02):
    """
    Optimize using repulsion energy as objective.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_energy = _repulsion_energy(pts)
    
    lr = lr_init
    momentum = np.zeros_like(pts)
    mu = 0.8
    
    for it in range(max_iters):
        n = len(pts)
        dots = pts @ pts.T
        np.fill_diagonal(dots, -1.0)
        
        # Gradient of repulsion energy
        max_dots = np.max(dots, axis=1)
        max_idx = np.argmax(dots, axis=1)
        
        # For each point, the nearest neighbor contributes to the energy
        # d(1/(1-max_dot))/d x_i = max_dot * (nearest_neighbor_dir - x_i) / (1-max_dot)^2
        nearest_neighbors = pts[max_idx]
        
        # Gradient contribution from nearest neighbors
        diff = nearest_neighbors - pts
        coeff = 1.0 / np.maximum(1e-10, 1.0 - max_dots)**2
        grad = coeff[:, np.newaxis] * diff
        
        # Project to tangent space
        tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Update
        momentum = mu * momentum - lr * tangent_grad
        new_pts = pts + momentum
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _repulsion_energy(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            lr *= 0.8
        
        if it % 20 == 0 and lr > 0.001:
            lr *= 0.95
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_pts = None
    best_cr = np.pi
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib)
    
    # Random
    for _ in range(5):
        pts_r = rng.standard_normal((N, 3))
        pts_r /= np.linalg.norm(pts_r, axis=1, keepdims=True)
        initializations.append(pts_r)
    
    # Perturbed Fibonacci
    for noise_scale in [0.05, 0.1, 0.2]:
        pts_pert = pts_fib + noise_scale * rng.standard_normal((N, 3))
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        initializations.append(pts_pert)
    
    # Different golden angle variants
    for angle_mult in [1.0, 1.5, 2.0]:
        i4 = np.arange(N) + 0.5
        z4 = 1.0 - 2.0 * i4 / N
        phi4 = np.pi * (1.0 + np.sqrt(5.0)) * i4 * angle_mult
        r4 = np.sqrt(np.maximum(0, 1.0 - z4 * z4))
        pts_ga = np.column_stack([r4 * np.cos(phi4), r4 * np.sin(phi4), z4])
        initializations.append(pts_ga)
    
    # Evaluate initializations
    for init_pts in initializations:
        if time.time() > deadline - 30:
            break
        cr = _covering_radius(init_pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = init_pts.copy()
    
    # Run optimization on each initialization
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # Phase 1: Repulsion-based optimization
        rep_pts, _ = _optimize_repulsion(init_pts, max_iters=200, lr_init=0.02)
        cr1 = _covering_radius(rep_pts)
        
        # Phase 2: Soft-max optimization
        soft_pts, _ = _soft_max_optimize(rep_pts, max_iters=300, beta=80.0)
        cr2 = _covering_radius(soft_pts)
        
        # Use the better result
        if cr2 < cr1:
            opt_pts = soft_pts
            opt_cr = cr2
        else:
            opt_pts = rep_pts
            opt_cr = cr1
        
        if opt_cr < best_cr:
            best_cr = opt_cr
            best_pts = opt_pts.copy()
        
        # If we have time, do additional fine-tuning on the best so far
        if time.time() < deadline - 15 and best_pts is not None:
            # Fine-tune with higher beta
            fine_pts, _ = _soft_max_optimize(best_pts, max_iters=200, beta=120.0)
            fine_cr = _covering_radius(fine_pts)
            if fine_cr < best_cr:
                best_cr = fine_cr
                best_pts = fine_pts.copy()
    
    if best_pts is None:
        best_pts = initializations[0]
    
    return best_pts

def _soft_max_optimize(points, max_iters=300, beta=80.0):
    """Alias for _optimize_soft_max with cleaner interface."""
    return _optimize_soft_max(points, max_iters=max_iters, beta=beta)
# EVOLVE-BLOCK-END
