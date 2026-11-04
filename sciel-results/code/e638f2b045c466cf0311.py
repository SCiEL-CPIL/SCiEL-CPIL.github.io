import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

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

    # Normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)

    valid = norms > 1e-10
    if not np.all(valid):
        # Degenerate facets, handle gracefully
        radii = np.full(len(simplices), np.pi)
        centers = np.zeros((len(simplices), 3))
        centers[~valid] = [0, 0, 1]
        return radii, centers, simplices

    u = n / norms[:, np.newaxis]

    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1

    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)

    return radii, u, simplices

def get_covering_radius(points):
    radii, _, _ = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def chebyshev_objective_and_grad(points, beta=100.0):
    """
    Spherical Chebyshev approximation objective:
    Minimize a weighted sum of squared deviations of facet circumradii from their mean.
    This enforces equiangularity and provides a smooth landscape.
    
    Returns:
        value: float - the objective value
        grad: array of shape (N, 3) - gradient in ambient space (project to tangent if needed)
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    # Mean radius
    r_mean = np.mean(radii)
    
    # Squared deviations
    deviations = radii - r_mean
    sq_devs = deviations ** 2
    
    # Use log-sum-exp for stability
    max_sq = np.max(sq_devs)
    if max_sq < 1e-15:
        return 0.0, np.zeros_like(points)
    
    scaled = beta * (sq_devs - max_sq)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # Objective value: sqrt(mean of weighted squared deviations)
    value = np.sqrt(np.sum(weights * sq_devs))
    
    # Gradient computation
    grad_points = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-12:
            continue
            
        i_idx, j_idx, k_idx = simplices[m]
        center = centers[m]
        r = radii[m]
        dev = r - r_mean
        
        sin_r = np.sin(r)
        if sin_r < 1e-6:
            continue
            
        # Gradient of squared deviation: 2 * dev * dr/dpoint
        # dr/dpoint = -(u · dp) / sin(r) projected to tangent
        # For the Chebyshev objective, we want to reduce large deviations
        coeff = weights[m] * 2.0 * dev / (sin_r * sin_r)
        
        for idx in [i_idx, j_idx, k_idx]:
            p = points[idx]
            # Project center to tangent space at point p
            tangent_center = center - np.dot(center, p) * p
            grad_points[idx] += coeff * tangent_center
    
    return value, grad_points

def smooth_covering_radius(points, temperature=0.01):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    """
    radii, _, _ = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    
    # Log-sum-exp approximation to max
    max_r = np.max(radii)
    shifted = (radii - max_r) / temperature
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_r + temperature * log_sum
    return smooth_max

def greedy_init(n_points, seed=0):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Generate many candidates
        candidates = np.random.randn(500, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute min distance to existing points
        existing = np.array(points)
        # Dot products
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

def polish(points, time_budget):
    """
    High-precision polish pass with multi-phase refinement using Chebyshev objective.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = get_covering_radius(points)
    
    # Phase 1: Gradient descent on Chebyshev objective
    beta = 50.0
    lr = 0.02
    
    for step in range(2000):
        if time.time() - start > time_budget * 0.4:
            break
            
        val, grad = chebyshev_objective_and_grad(points, beta)
        
        radii, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
            
            # Adaptive beta: increase when spread is small
            spread = np.max(radii) - np.min(radii)
            if spread < 0.05:
                beta *= 1.02
            elif spread > 0.2:
                beta *= 0.98
            beta = np.clip(beta, 10.0, 500.0)
            
            # Project gradient to tangent space
            grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
            
            # Step with line search
            points_new = points + lr * grad_tangent
            norms = np.linalg.norm(points_new, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points_new /= norms
            
            # Check improvement
            new_radii, _, _ = compute_facet_data(points_new)
            if len(new_radii) > 0:
                new_val, _ = chebyshev_objective_and_grad(points_new, beta)
                if new_val < val * 1.01:
                    points = points_new
                    lr *= 1.05
                else:
                    lr *= 0.8
            else:
                lr *= 0.8
            
            if lr < 1e-6:
                break
    
    # Phase 2: Nelder-Mead refinement on worst facets
    if time.time() - start < time_budget * 0.7:
        radii, _, simplices = compute_facet_data(best_local)
        if len(radii) > 0:
            n_worst = max(5, len(radii) // 5)
            worst_facets = np.argsort(radii)[-n_worst:]
            worst_indices = set()
            for f in worst_facets:
                worst_indices.update(simplices[f].tolist())
            worst_indices = list(worst_indices)
            
            if worst_indices:
                remaining = time_budget * 0.7 - (time.time() - start) - 1.0
                if remaining > 1.0:
                    start_nm = time.time()
                    initial_radius = get_covering_radius(best_local)
                    x0 = best_local[worst_indices].flatten()
                    
                    def nm_objective(x_flat):
                        if time.time() - start_nm > remaining:
                            return initial_radius
                        new_points = best_local.copy()
                        new_points[worst_indices] = x_flat.reshape(-1, 3)
                        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                        return smooth_covering_radius(new_points, 0.01)
                    
                    try:
                        res = minimize(nm_objective, x0, method='Nelder-Mead', 
                                       options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                        new_points = best_local.copy()
                        new_points[worst_indices] = res.x.reshape(-1, 3)
                        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                        exact_radius = get_covering_radius(new_points)
                        if exact_radius < best_local_radius:
                            best_local = new_points
                            best_local_radius = exact_radius
                    except Exception:
                        pass
    
    # Phase 3: Final fine-tuning with very small steps on all facets
    if time.time() - start < time_budget - 1.0:
        radii, centers, simplices = compute_facet_data(best_local)
        if len(radii) > 0:
            for step in range(300):
                if time.time() - start > time_budget - 0.5:
                    break
                    
                radii, centers, simplices = compute_facet_data(best_local)
                if len(radii) == 0:
                    break
                    
                max_radius = np.max(radii)
                if max_radius < best_local_radius:
                    best_local_radius = max_radius
                    
                # Move all vertices slightly toward their facet centers
                updates = np.zeros_like(best_local)
                for m in range(len(radii)):
                    simplex = simplices[m]
                    i_idx, j_idx, k_idx = simplex
                    center = centers[m]
                    step_size = 0.001 * radii[m]
                    updates[i_idx] += step_size * (center - best_local[i_idx])
                    updates[j_idx] += step_size * (center - best_local[j_idx])
                    updates[k_idx] += step_size * (center - best_local[k_idx])
                
                best_local = best_local + updates
                best_local /= np.linalg.norm(best_local, axis=1, keepdims=True)
                
                # Check if we improved
                new_radii, _, _ = compute_facet_data(best_local)
                if len(new_radii) > 0:
                    new_max = np.max(new_radii)
                    if new_max > best_local_radius + 1e-8:
                        break
                    else:
                        best_local_radius = new_max
    
    return best_local

def construct():
    start_time = time.time()
    time_limit = 145.0
    
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
    
    # 2. Random points
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(init_rand)
    
    # 3. Greedy initialization
    init_greedy = greedy_init(N, seed=42)
    initializations.append(init_greedy)
    
    # 4. Another greedy with different seed
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(init_greedy2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # More randomized restarts with varied noise levels
    for s in range(15):
        np.random.seed(1000 + s * 7)
        noise_scale = 0.05 + 0.15 * (s / 15)
        init = init_fib + np.random.randn(N, 3) * noise_scale
        init /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
        initializations.append(init)
    
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            np.random.seed(restart * 137)
            points = init_fib + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Local Optimization using Chebyshev objective
        alpha = 0.03
        no_improve = 0
        stuck_counter = 0
        
        for step in range(3000):
            if time.time() - start_time > time_limit:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            
            if len(radii) == 0:
                break
                
            max_radius = np.max(radii)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve = 0
                stuck_counter = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    alpha *= 0.92
                    if alpha < 1e-5:
                        break
                if no_improve > 150:
                    # Try a small random perturbation to escape local minimum
                    np.random.seed(step * 31 + restart * 17)
                    perturbation = np.random.randn(N, 3) * 0.01
                    points = points + perturbation
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    no_improve = 0
                    stuck_counter += 1
                    if stuck_counter > 5:
                        break
            
            # Use Chebyshev gradient for updates
            val, grad = chebyshev_objective_and_grad(points, beta=50.0)
            
            # Project to tangent space
            grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
            
            # Combine with facet push for worst facets
            n_worst = max(5, len(radii) // 7)
            worst_indices = np.argsort(radii)[-n_worst:]
            mask = np.zeros(len(radii), dtype=bool)
            mask[worst_indices] = True
            
            updates = np.zeros_like(points)
            for m in np.where(mask)[0]:
                simplex = simplices[m]
                i_idx, j_idx, k_idx = simplex
                center = centers[m]
                step_size = alpha * radii[m]
                updates[i_idx] += step_size * (center - points[i_idx])
                updates[j_idx] += step_size * (center - points[j_idx])
                updates[k_idx] += step_size * (center - points[k_idx])
            
            # Blend Chebyshev gradient with facet push
            points = points + 0.5 * grad_tangent + updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            # Guard against coincident points
            min_dist = np.min(np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2))
            if min_dist < 0.01:
                idx = np.argmin(np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2))
                i, j = divmod(idx, N)
                if i != j:
                    direction = points[j] - points[i]
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-10:
                        direction /= norm_dir
                        points[i] -= 0.02 * direction
                        points[j] += 0.02 * direction
                        points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Final polish pass on the best configuration found
    if best_points is not None and time.time() - start_time < time_limit - 2.0:
        remaining_time = time_limit - 2.0 - (time.time() - start_time)
        if remaining_time > 2.0:
            polished = polish(best_points, remaining_time)
            # Check if polished is better
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                pol_radius = np.max(radii)
                if pol_radius < best_radius:
                    best_radius = pol_radius
                    best_points = polished
    
    if best_points is None:
        best_points = init_fib
        
    # Final safety check: ensure no coincident points
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-6):
        best_points = init_fib
    
    return best_points
