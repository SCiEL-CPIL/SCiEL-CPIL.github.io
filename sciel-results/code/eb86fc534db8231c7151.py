import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_facet_data(points):
    """
    Computes the circumradius and circumcenter direction for each facet of the convex hull.
    Returns:
        radii: array of shape (M,)
        centers: array of shape (M, 3) unit vectors
        simplices: array of shape (M, 3)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    
    # Initialize outputs
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    centers[~valid] = [0, 0, 1]
    
    if not np.all(valid):
        if np.any(valid):
            a_v = a[valid]
            b_v = b[valid]
            c_v = c[valid]
            n_v = n[valid]
            norms_v = norms[valid]
            
            u = n_v / norms_v[:, np.newaxis]
            centroids = (a_v + b_v + c_v) / 3.0
            dots = np.sum(u * centroids, axis=1)
            u[dots < 0] *= -1
            
            cos_theta = np.sum(u * a_v, axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii[valid] = np.arccos(cos_theta)
            centers[valid] = u
            
        return radii, centers, simplices

    # All valid
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def greedy_init(n_points, seed=42):
    """
    Greedy initialization: Farthest Point Sampling.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        candidates = np.random.randn(1000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        existing = np.array(points)
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
        
    return np.array(points)

def lloyd_step(points, alpha=0.5):
    """
    Perform one step of Lloyd's algorithm.
    Move each point towards the centroid of its Voronoi cell.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points

    radii, centers, simplices = compute_facet_data(points)
    
    if len(radii) == 0:
        return points

    updates = np.zeros_like(points)
    counts = np.zeros(N)
    
    for m in range(len(centers)):
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        updates[i_idx] += centers[m]
        updates[j_idx] += centers[m]
        updates[k_idx] += centers[m]
        counts[i_idx] += 1
        counts[j_idx] += 1
        counts[k_idx] += 1
    
    valid = counts > 0
    updates[valid] /= counts[valid, np.newaxis]
    
    new_points = points.copy()
    new_points[valid] = points[valid] + alpha * (updates[valid] - points[valid])
    
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def facet_push_step(points, radii, centers, simplices, T=0.1, percentile=80):
    """
    Push vertices of large facets towards the circumcenter.
    """
    if len(radii) == 0:
        return points
        
    threshold = np.percentile(radii, percentile)
    mask = radii > threshold
    
    if not np.any(mask):
        max_idx = np.argmax(radii)
        mask = np.zeros(len(radii), dtype=bool)
        mask[max_idx] = True
        
    updates = np.zeros_like(points)
    
    for m in np.where(mask)[0]:
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        center = centers[m]
        radius = radii[m]
        
        step = T * radius
        
        updates[i_idx] += step * (center - points[i_idx])
        updates[j_idx] += step * (center - points[j_idx])
        updates[k_idx] += step * (center - points[k_idx])
        
    new_points = points + updates
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def smooth_objective_and_grad(points):
    """
    Compute a smooth surrogate for the max facet radius and its gradient.
    Uses log-sum-exp of squared radii with proper gradient derivation.
    
    For a facet with vertices a, b, c and circumcenter u (unit vector),
    the circumradius r satisfies cos(r) = u·a = u·b = u·c.
    
    Differentiating: -sin(r) dr = du·a + u·da
    Since u is the circumcenter, du/da = -a/sin(r) (for the component along a).
    
    The gradient of r w.r.t. point a is:
    dr/da = -(u·da) / sin(r) + (da·a) / sin(r) ... 
    Actually, for a point on the sphere, the gradient in the tangent space is:
    grad_r = -u / sin(r) (projected to tangent space)
    
    For the smooth max surrogate f = sqrt(logsumexp(beta * r_i^2) / beta):
    df/da = sum_i w_i * 2*beta*r_i * (dr_i/da) / (2*sqrt(beta*logsumexp(beta*r_i^2)))
         = sum_i w_i * beta*r_i * (-u_i/sin(r_i)) / (value*beta)
         = -sum_i w_i * r_i / (sin(r_i) * value) * u_i
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    beta = 100.0
    
    max_r2 = np.max(radii**2)
    scaled = beta * (radii**2 - max_r2)
    
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    log_sum_exp_val = beta * max_r2 + max_scaled + np.log(sum_exp)
    
    value = np.sqrt(log_sum_exp_val / beta)
    
    grad_points = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-12:
            continue
            
        i_idx, j_idx, k_idx = simplices[m]
        center = centers[m]
        r = radii[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-6:
            continue
            
        # Gradient contribution: -w_m * r_m / (sin(r_m) * value) * center_m
        coeff = -weights[m] * r / (sin_r * value)
        
        grad_points[i_idx] += coeff * center
        grad_points[j_idx] += coeff * center
        grad_points[k_idx] += coeff * center
        
    return value, grad_points

def polish(points, time_budget, start_alpha=0.05):
    """
    High-precision polish pass.
    """
    start = time.time()
    alpha = start_alpha
    best_local = points.copy()
    best_local_radius = np.pi
    
    for step in range(1000):
        if time.time() - start > time_budget * 0.5:
            break
            
        val, grad = smooth_objective_and_grad(points)
        
        radii, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
                
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        points = points + alpha * grad_tangent
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha *= 0.99
        
        if alpha < 1e-5:
            break
            
    points = best_local
    
    alpha_push = 0.02
    for step in range(2000):
        if time.time() - start > time_budget:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
            
        max_r = np.max(radii)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
            
        threshold = np.percentile(radii, 90)
        mask = radii > threshold
        
        if not np.any(mask):
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
            
        updates = np.zeros_like(points)
        
        for m in np.where(mask)[0]:
            idx = simplices[m]
            i_idx, j_idx, k_idx = idx
            center = centers[m]
            r = radii[m]
            
            step_size = alpha_push * r
            
            updates[i_idx] += step_size * (center - points[i_idx])
            updates[j_idx] += step_size * (center - points[j_idx])
            updates[k_idx] += step_size * (center - points[k_idx])
            
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha_push *= 0.995
        
        if alpha_push < 1e-5:
            break
            
    return best_local

def spectral_refinement(points, time_budget):
    """
    Structural change: Spherical Linear Programming (SLP) dual-inspired refinement.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    
    n_grid = 2000
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    sigma = 0.1
    
    for step in range(500):
        if time.time() - start > time_budget:
            break
        
        dots = grid_points @ points.T
        dist_sq = 2.0 - 2.0 * dots
        dist_sq = np.maximum(dist_sq, 0.0)
        
        weights = np.exp(-dist_sq / (2.0 * sigma**2))
        potential = np.sum(weights, axis=1)
        
        radii, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
        
        threshold = np.percentile(potential, 20)
        low_mask = potential < threshold
        
        if not np.any(low_mask):
            break
            
        low_points = grid_points[low_mask]
        
        updates = np.zeros_like(points)
        
        for i in range(N):
            dists = np.linalg.norm(low_points - points[i], axis=1)
            nearest_idx = np.argmin(dists)
            target = low_points[nearest_idx]
            
            step_size = 0.01
            updates[i] = step_size * (target - points[i])
        
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        sigma *= 0.99
        
        if sigma < 0.01:
            break
    
    return best_local

def nelder_mead_refine(points, time_budget, max_iter=2000):
    """
    Nelder-Mead simplex method refinement on the sphere.
    This is a derivative-free method that can escape some local minima.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = np.pi
    
    radii, _, _ = compute_facet_data(points)
    if len(radii) > 0:
        best_radius = np.max(radii)
    
    # Create initial simplex
    # We perturb the best point to create a simplex in the tangent space
    # For simplicity, we use a small perturbation in a random direction
    np.random.seed(42)
    
    # We'll do a simplified version: random restarts with small perturbations
    # and keep the best
    current = points.copy()
    
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Generate a random perturbation
        # The perturbation size decays over time
        scale = 0.05 * (0.95 ** (i / 50.0))
        if scale < 0.001:
            scale = 0.001
            
        perturbation = np.random.randn(N, 3) * scale
        # Project to tangent space
        perturbation -= np.sum(perturbation * current, axis=1, keepdims=True) * current
        new_points = current + perturbation
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        radii, _, _ = compute_facet_data(new_points)
        if len(radii) == 0:
            continue
            
        new_radius = np.max(radii)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            current = new_points.copy()
        elif new_radius < best_radius * 1.01:
            # Accept with some probability to escape local minima
            if np.random.rand() < 0.1:
                current = new_points.copy()
        else:
            # Reject, but sometimes accept to explore
            if np.random.rand() < 0.05:
                current = new_points.copy()
                
    return best_points

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    initializations = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(init_fib)
    
    # 2. Greedy Initialization
    init_greedy = greedy_init(N, seed=10)
    initializations.append(init_greedy)
    
    # 3. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(init_rand)
    
    # 4. Greedy 2
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(init_greedy2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # 6. Greedy 3
    init_greedy3 = greedy_init(N, seed=999)
    initializations.append(init_greedy3)
    
    # 7. Fibonacci with different phase
    i2 = np.arange(N) + 0.5
    z2 = 1.0 - 2.0 * i2 / N
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + 0.3)
    r2 = np.sqrt(1.0 - z2 * z2)
    init_fib2 = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
    initializations.append(init_fib2)
    
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Perturb the best found so far
            if best_points is not None:
                np.random.seed(restart * 137)
                points = best_points + np.random.randn(N, 3) * 0.05
            else:
                points = init_greedy + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        # Optimization loop
        T = 0.05
        alpha_lloyd = 0.5
        no_improve_count = 0
        current_radius = np.pi
        
        # Inner loop
        while True:
            if time.time() - start_time > time_limit:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            
            if len(radii) == 0:
                break
                
            max_radius = np.max(radii)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    T *= 0.9
                    alpha_lloyd *= 0.95
                if no_improve_count > 100:
                    break
                if T < 1e-6 and alpha_lloyd < 1e-6:
                    break
            
            # Alternate steps
            if no_improve_count % 10 == 0:
                points = lloyd_step(points, alpha=alpha_lloyd)
            else:
                points = facet_push_step(points, radii, centers, simplices, T=T, percentile=80)
                
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            T *= 0.995
            alpha_lloyd *= 0.995
            
        # Nelder-Mead style refinement
        if time.time() - start_time < time_limit - 15.0:
            nm_time = min(10.0, time_limit - (time.time() - start_time))
            refined = nelder_mead_refine(points, nm_time)
            radii, _, _ = compute_facet_data(refined)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = refined.copy()
                    
        # Spectral refinement
        if time.time() - start_time < time_limit - 10.0:
            polished = spectral_refinement(points, time_budget=min(10.0, time_limit - (time.time() - start_time)))
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = polished
    
    # Final Polish
    if best_points is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5.0:
            polished = polish(best_points, remaining_time)
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = polished
    else:
        best_points = init_fib
        
    return best_points
