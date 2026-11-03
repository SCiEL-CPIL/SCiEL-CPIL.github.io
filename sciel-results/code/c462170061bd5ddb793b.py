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

def get_objective_and_gradient(points, tau=0.05):
    """
    Computes a smooth surrogate for the covering radius using log-sum-exp.
    Returns the objective value and the negative gradient (descent direction).
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi, np.zeros_like(points)
        
    # Log-sum-exp for numerical stability: approximates max(radii) smoothly
    r_scaled = radii / tau
    r_max = np.max(r_scaled)
    exp_terms = np.exp(r_scaled - r_max)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Objective: tau * log(sum(exp(radii/tau))) approximates max(radii)
    objective = tau * (np.log(sum_exp) + r_max)
    
    # Gradient: d/dp_i [tau * log(sum_m exp(r_m/tau))] 
    # = sum_m weights[m] * d(r_m)/d(p_i)
    # For facet m with vertices i,j,k and circumcenter c_m:
    # d(r_m)/d(p_i) = -c_m / sin(r_m) (projected to tangent space)
    # So the descent direction (negative gradient) is:
    # -sum_m weights[m] * (-c_m / sin(r_m)) = sum_m weights[m] * c_m / sin(r_m)
    
    updates = np.zeros_like(points)
    sin_r = np.sin(radii)
    sin_r[sin_r < 1e-6] = 1e-6
    
    # Vectorized update computation
    for m in range(len(radii)):
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        center = centers[m]
        w = weights[m]
        factor = w / sin_r[m]
        
        updates[i_idx] += factor * center
        updates[j_idx] += factor * center
        updates[k_idx] += factor * center
        
    # Project updates to tangent plane (remove radial component)
    for i in range(N):
        proj = np.dot(updates[i], points[i])
        updates[i] -= proj * points[i]
        
    return objective, updates

def greedy_init(n_points, seed=42):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = []
    
    # First point: random
    p = np.random.randn(3)
    p /= np.linalg.norm(p)
    points.append(p)
    
    for _ in range(n_points - 1):
        # Generate many candidate points
        n_candidates = 5000
        candidates = np.random.randn(n_candidates, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Compute all distances at once
        pts_arr = np.array(points)
        cos_dists = candidates @ pts_arr.T  # (n_candidates, n_points)
        min_cos = np.min(cos_dists, axis=1)
        # We want to maximize the minimum angular distance, which means minimizing min_cos
        # Actually, we want the point that is farthest from all existing points
        # So we want to maximize the minimum angular distance = minimize min_cos
        # Wait, cos is decreasing with angle, so min_cos corresponds to max angle
        # We want the candidate with the LARGEST minimum angle, i.e., SMALLEST min_cos
        best_idx = np.argmin(min_cos)
        best_candidate = candidates[best_idx]
        
        points.append(best_candidate)
    
    return np.array(points)

def force_init(n_points, iterations=100, seed=42):
    """
    Initialize points using a repulsive force model to spread them out.
    """
    np.random.seed(seed)
    points = np.random.randn(n_points, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    for _ in range(iterations):
        forces = np.zeros_like(points)
        for i in range(n_points):
            f = 0
            for j in range(n_points):
                if i == j: continue
                diff = points[i] - points[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-10:
                    f += np.random.randn(3)
                else:
                    f += diff / (d2 * np.sqrt(d2))
            forces[i] = f
            
        points += 0.01 * forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
    return points

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin
    
    best_points = None
    best_radius = np.pi
    best_obj = np.pi
    
    # Initial configurations
    initializations = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(init_fib)
    
    # 2. Greedy initialization
    init_greedy = greedy_init(N, seed=42)
    initializations.append(init_greedy)
    
    # 3. Force-based initialization
    init_force = force_init(N, iterations=50, seed=42)
    initializations.append(init_force)
    
    # 4. Force-based with different seed
    init_force2 = force_init(N, iterations=50, seed=123)
    initializations.append(init_force2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # 6. More randomized restarts with varying noise levels
    for s in range(10):
        np.random.seed(1000 + s * 13)
        noise_scale = 0.05 + 0.15 * s / 10
        init = init_fib + np.random.randn(N, 3) * noise_scale
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
        
    max_restarts = len(initializations)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        points = initializations[restart].copy()
        
        # Phase 1: Gradient Descent with adaptive tau and line search
        tau = 0.08
        alpha = 0.05
        no_improve = 0
        prev_obj = np.inf
        
        for step in range(3000):
            if time.time() - start_time > time_limit - 10.0:
                break
                
            obj, grad = get_objective_and_gradient(points, tau)
            
            # Check actual covering radius for best tracking
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) > 0:
                max_radius = np.max(radii)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 200:
                        break
            else:
                break
            
            # Adapt tau: decrease slowly to sharpen the approximation
            tau = max(0.02, tau * 0.999)
            
            direction = grad  # Descent direction
            
            # Normalize direction for consistent step size
            norm_dir = np.linalg.norm(direction, axis=1)
            norm_dir[norm_dir < 1e-10] = 1e-10
            direction = direction / norm_dir[:, np.newaxis]
            
            # Backtracking line search
            step_size = alpha
            improved = False
            for _ in range(8):
                new_points = points + step_size * direction
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_obj, _ = get_objective_and_gradient(new_points, tau)
                
                if new_obj < obj - 1e-12:
                    points = new_points
                    improved = True
                    break
                else:
                    step_size *= 0.5
                    
            if not improved:
                alpha *= 0.7
                if alpha < 1e-5:
                    break
                    
            # Periodic random perturbation to escape local minima
            if step % 300 == 299:
                np.random.seed(int(time.time() * 1000) % 100000)
                perturb = np.random.randn(N, 3) * 0.015
                points += perturb
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                    
        # Phase 2: High-precision polish focusing on worst facets
        if time.time() - start_time < time_limit - 5.0:
            alpha_polish = 0.02
            for step in range(1500):
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                radii, centers, simplices = compute_facet_data(points)
                if len(radii) == 0:
                    break
                
                max_radius = np.max(radii)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                
                # Focus on worst facets - top 15%
                if len(radii) > 5:
                    threshold = np.percentile(radii, 85)
                    mask = radii > threshold
                else:
                    max_idx = np.argmax(radii)
                    mask = np.zeros(len(radii), dtype=bool)
                    mask[max_idx] = True
                    
                if not np.any(mask):
                    break
                    
                updates = np.zeros_like(points)
                
                for m in np.where(mask)[0]:
                    simplex = simplices[m]
                    i_idx, j_idx, k_idx = simplex
                    
                    center = centers[m]
                    # Move vertices towards the circumcenter direction
                    # Use a step proportional to the radius to make larger moves for worse facets
                    step_size = alpha_polish * min(radii[m], 0.3)
                    
                    # Move each vertex towards the circumcenter
                    for idx in [i_idx, j_idx, k_idx]:
                        diff = center - points[idx]
                        # Project to tangent plane
                        proj = np.dot(diff, points[idx])
                        diff_tangent = diff - proj * points[idx]
                        updates[idx] += step_size * diff_tangent
                    
                # Project to tangent and normalize
                for i in range(N):
                    proj = np.dot(updates[i], points[i])
                    updates[i] -= proj * points[i]
                
                points += updates
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
                alpha_polish *= 0.992

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
