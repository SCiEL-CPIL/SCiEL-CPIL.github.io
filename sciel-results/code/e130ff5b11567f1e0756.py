"""
Structural change: reformulate as a soft-max repulsion problem on the sphere.
Instead of hull-facet circumcenter repair, we use a smooth soft-max penalty on
the closest neighbors to create uniform repulsive pressure, combined with
projected gradient ascent on the minimum pairwise dot product.
"""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    facets = hull.simplices
    if len(facets) == 0:
        return np.pi
    
    min_cos = 1.0
    for facet in facets:
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_val = p @ a
        if cos_val < min_cos:
            min_cos = cos_val
    
    return np.arccos(np.clip(min_cos, -1.0, 1.0))


def _soft_max_repulsion(points, temperature=0.01, k_nearest=8):
    """
    Compute a smooth approximation to the minimum pairwise distance using
    soft-max over the k_nearest closest neighbors for each point.
    Returns (value, gradient) where value is the soft-min dot product
    (lower = better separation) and gradient is w.r.t. the flat vector.
    """
    n = points.shape[0]
    # Compute all pairwise dot products
    dot = points @ points.T
    np.fill_diagonal(dot, 1.0)  # self-dot is 1
    
    # For each point, find k_nearest closest neighbors (highest dot product)
    # We want to maximize the minimum dot product (minimize max angular distance)
    # Actually, we want to maximize the minimum angular separation = maximize min arccos(dot)
    # = minimize max dot product among pairs
    # So we want to minimize the soft-max of dot products
    
    # For efficiency, use soft-max over all other points with temperature
    # This gives a smooth approximation to max_j dot[i,j]
    
    # Soft-max: for each i, compute exp(dot[i,j]/T) / sum_j exp(dot[i,j]/T)
    # Then value = sum_i T * log(sum_j exp(dot[i,j]/T))
    # This is the soft-max of dot products
    
    # To make it focus on the closest neighbors, use a lower temperature
    # and subtract the max to avoid overflow
    dot_no_diag = dot.copy()
    np.fill_diagonal(dot_no_diag, -1e10)  # exclude self
    
    max_dot = np.max(dot_no_diag, axis=1, keepdims=True)
    exp_dot = np.exp((dot_no_diag - max_dot) / temperature)
    log_sum_exp = np.log(np.sum(exp_dot, axis=1, keepdims=True)) + max_dot / temperature
    
    # Soft-max value (approximation to max dot product per point)
    soft_max_vals = log_sum_exp  # shape (n, 1)
    value = np.mean(soft_max_vals)
    
    # Gradient: d/dx_i [ T * log(sum_j exp(dot[i,j]/T)) ]
    # = T * (1/T) * sum_j exp(dot[i,j]/T) / sum_j exp(dot[i,j]/T) * d dot[i,j]/dx_i
    # = sum_j softmax_weight[i,j] * d dot[i,j]/dx_i
    # d dot[i,j]/dx_i = x_j (for i != j)
    
    softmax_weights = exp_dot / np.sum(exp_dot, axis=1, keepdims=True)  # shape (n, n)
    
    # Gradient w.r.t. points[i] = sum_j softmax_weights[i,j] * points[j]
    grad_points = softmax_weights @ points  # shape (n, 3)
    
    # Total gradient is the sum over all i
    grad = grad_points  # Each row is the gradient contribution for that point
    
    return value, grad


def _project_to_sphere(points):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return points / norms


def _soft_max_optimize(points, deadline, max_iters=5000, base_lr=0.01, 
                        temperature_start=0.05, temperature_end=0.005):
    """
    Optimize using projected gradient descent on the soft-max repulsion objective.
    We minimize the soft-max of pairwise dot products (which approximates the
    maximum dot product = minimum angular separation).
    """
    points = points.copy().astype(np.float64)
    best_value = _covering_radius(points)
    best_points = points.copy()
    
    lr = base_lr
    patience = 0
    max_patience = 200
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Anneal temperature
        progress = it / max(max_iters - 1, 1)
        temperature = temperature_start + (temperature_end - temperature_start) * progress
        
        # Compute soft-max objective and gradient
        value, grad = _soft_max_repulsion(points, temperature=temperature)
        
        # Project gradient onto tangent space
        # grad_tangent = grad - (grad . x_i) * x_i
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Gradient descent step
        new_points = points - lr * grad_tangent
        new_points = _project_to_sphere(new_points)
        
        # Evaluate actual covering radius (expensive, do less frequently)
        if it % 50 == 0 or it == max_iters - 1:
            cr = _covering_radius(new_points)
            if cr < best_value:
                best_value = cr
                best_points = new_points.copy()
                patience = 0
                lr = min(lr * 1.1, 0.05)
            else:
                patience += 1
                lr *= 0.95
            
            if patience > max_patience:
                # Restart with perturbation
                perturbation = 0.1 * np.random.default_rng(it).standard_normal((N, 3))
                new_points = best_points + perturbation
                new_points = _project_to_sphere(new_points)
                patience = 0
                lr = base_lr
        
        points = new_points
    
    return best_points, best_value


def _repulsion_force_optimize(points, deadline, max_iters=3000, base_lr=0.005,
                               k_nearest=6, force_strength=1.0):
    """
    Alternative: direct repulsion force model.
    Each point is pushed away from its k_nearest neighbors with a force
    proportional to how close they are. This creates uniform pressure.
    """
    points = points.copy().astype(np.float64)
    best_value = _covering_radius(points)
    best_points = points.copy()
    
    lr = base_lr
    patience = 0
    max_patience = 150
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Compute pairwise dot products
        dot = points @ points.T
        np.fill_diagonal(dot, -2.0)  # exclude self, mark as far away
        
        # For each point, find k_nearest neighbors (highest dot)
        # and compute repulsive forces
        forces = np.zeros_like(points)
        
        for i in range(N):
            # Get dot products with all other points
            dots_i = dot[i].copy()
            
            # Find k_nearest (highest dot values, excluding self)
            # Use partition for efficiency
            if k_nearest < N - 1:
                # Get indices of k_nearest largest values
                idx = np.argpartition(-dots_i, k_nearest)[:k_nearest]
            else:
                idx = np.arange(N)
                idx = idx[idx != i]
            
            # Compute forces from these neighbors
            for j in idx:
                if j == i:
                    continue
                d = dot[i, j]
                # Force magnitude: stronger when closer (higher dot)
                # Use exponential decay to focus on close neighbors
                weight = force_strength * np.exp(-d * 5.0)
                # Force direction: away from neighbor j
                # On sphere, the tangent direction away from j
                tangent = -points[j] + np.dot(points[j], points[i]) * points[i]
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    forces[i] += weight * tangent / norm_t
        
        # Project forces to tangent space (they already are, but ensure)
        forces = forces - np.sum(forces * points, axis=1, keepdims=True) * points
        
        # Update
        new_points = points + lr * forces
        new_points = _project_to_sphere(new_points)
        
        # Evaluate
        if it % 30 == 0 or it == max_iters - 1:
            cr = _covering_radius(new_points)
            if cr < best_value:
                best_value = cr
                best_points = new_points.copy()
                patience = 0
                lr = min(lr * 1.05, 0.02)
            else:
                patience += 1
                lr *= 0.95
            
            if patience > max_patience:
                # Perturb and restart
                rng = np.random.default_rng(it + 1000)
                perturbation = 0.2 * rng.standard_normal((N, 3))
                new_points = best_points + perturbation
                new_points = _project_to_sphere(new_points)
                patience = 0
                lr = base_lr
        
        points = new_points
    
    return best_points, best_value


def _fibonacci_spiral(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Phase 1: Start with Fibonacci spiral and optimize with soft-max
    fib_points = _fibonacci_spiral(N)
    points, radius = _soft_max_optimize(fib_points, deadline, max_iters=3000, 
                                          base_lr=0.008, temperature_start=0.04,
                                          temperature_end=0.003)
    best_points = points
    best_radius = radius
    
    # Phase 2: Repulsion force optimization from best
    if time.time() < deadline - 40.0:
        points, radius = _repulsion_force_optimize(best_points, deadline, max_iters=2000,
                                                     base_lr=0.003, k_nearest=6,
                                                     force_strength=0.8)
        if radius < best_radius:
            best_radius = radius
            best_points = points
    
    # Phase 3: Random restarts with both methods
    num_restarts = 4
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 20.0:
            break
        
        # Alternate between perturbing best and random init
        if restart % 2 == 0:
            perturbation = 0.12 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = _project_to_sphere(points)
        else:
            points = rng.standard_normal((N, 3))
            points = _project_to_sphere(points)
        
        # Use soft-max optimization
        max_iters = min(2000, int(remaining * 15))
        optimized, radius = _soft_max_optimize(points, deadline, max_iters=max_iters,
                                                 base_lr=0.006, temperature_start=0.03,
                                                 temperature_end=0.002)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Phase 4: Final polish with very small steps
    if best_points is not None and time.time() < deadline - 10.0:
        points, radius = _soft_max_optimize(best_points, deadline, max_iters=1500,
                                              base_lr=0.002, temperature_start=0.01,
                                              temperature_end=0.001)
        if radius < best_radius:
            best_radius = radius
            best_points = points
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
