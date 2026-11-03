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

def get_objective_and_gradient(points, tau=0.1):
    """
    Computes the log-sum-exp objective and its gradient.
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi, np.zeros_like(points)
        
    # Log-sum-exp for numerical stability
    r_scaled = radii / tau
    r_max = np.max(r_scaled)
    exp_terms = np.exp(r_scaled - r_max)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    objective = np.log(sum_exp) + r_max
    
    # Gradient computation
    updates = np.zeros_like(points)
    sin_r = np.sin(radii)
    sin_r[sin_r < 1e-6] = 1e-6
    
    for m in range(len(radii)):
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        center = centers[m]
        w = weights[m]
        factor = w / sin_r[m]
        
        updates[i_idx] += factor * center
        updates[j_idx] += factor * center
        updates[k_idx] += factor * center
        
    # Project updates to tangent plane
    for i in range(N):
        proj = np.dot(updates[i], points[i])
        updates[i] -= proj * points[i]
        
    return objective, updates

def force_init(n_points, iterations=100, seed=42):
    """
    Initialize points using a repulsive force model to spread them out.
    """
    np.random.seed(seed)
    points = np.random.randn(n_points, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Simple repulsive force: minimize sum of 1/d^2 or similar
    # We just do gradient descent on sum of distances
    for _ in range(iterations):
        # Compute pairwise distances
        # This is O(N^2) which is fine for N=130
        dists = 1.0 - points @ points.T
        # Avoid self-interaction and division by zero
        np.fill_diagonal(dists, np.inf)
        # Force is proportional to 1/d^2 * (p_i - p_j)
        # We want to maximize minimum distance, so we minimize sum of 1/d^2
        # Gradient of 1/d^2 wrt p_i is -2/d^3 * (p_i - p_j)
        # So we move p_i in direction sum_j (p_i - p_j)/d^3
        forces = np.zeros_like(points)
        # Vectorized computation of forces is tricky with inf, so loop or mask
        mask = np.isfinite(dists)
        # d = sqrt(2(1-cos)) is not needed if we use vector diff
        # Let's just do a simple loop for clarity and safety with N=130
        for i in range(n_points):
            f = 0
            for j in range(n_points):
                if i == j: continue
                diff = points[i] - points[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-10:
                    # Random kick
                    f += np.random.randn(3)
                else:
                    f += diff / (d2 * np.sqrt(d2)) # 1/d^3 direction
            forces[i] = f
            
        # Update
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
    
    # 2. Force-based initialization
    init_force = force_init(N, iterations=50, seed=42)
    initializations.append(init_force)
    
    # 3. Force-based with different seed
    init_force2 = force_init(N, iterations=50, seed=123)
    initializations.append(init_force2)
    
    # 4. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # 5. More randomized restarts
    for s in range(10):
        np.random.seed(1000 + s * 7)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.1 * s / 10)
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
        
    max_restarts = len(initializations)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        points = initializations[restart].copy()
        
        # Phase 1: Gradient Descent with Line Search
        tau = 0.1
        alpha = 0.1
        no_improve = 0
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
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
                    if no_improve > 100:
                        break
            else:
                break
            
            # Line Search
            # Try to find a step size that reduces the objective
            # We use the objective from log-sum-exp as a proxy for speed,
            # but we want to minimize the true max radius.
            # Let's use the true max radius for the line search decision if possible,
            # but it's expensive. Let's stick to the smooth objective for the search,
            # but verify with the true radius.
            
            # Gradient direction is -grad (descent)
            # But we already computed grad as the direction to ADD to points?
            # In get_objective_and_gradient, updates are the gradient of the objective.
            # Descent direction is -updates.
            # Wait, in the previous code:
            # "Descent: p = p - alpha * grad J"
            # "updates" was calculated as sum w * u / sin.
            # grad r_f wrt p_i = -u_f / sin(r_f).
            # So grad J = - sum w * u / sin.
            # So -grad J = + sum w * u / sin.
            # So `updates` in the function is actually -grad J (the descent direction).
            # Let's verify:
            # If we move p in direction u, cos_theta increases, r decreases.
            # So moving towards u reduces r.
            # So yes, `updates` is the descent direction.
            
            direction = updates = grad # The function returns the descent direction
            
            # Normalize direction to have unit norm for consistent step size
            norm_dir = np.linalg.norm(direction, axis=1)
            norm_dir[norm_dir < 1e-10] = 1e-10
            direction = direction / norm_dir[:, np.newaxis]
            
            # Backtracking line search
            step_size = alpha
            improved = False
            for _ in range(10):
                new_points = points + step_size * direction
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_obj, _ = get_objective_and_gradient(new_points, tau)
                
                if new_obj < obj:
                    points = new_points
                    improved = True
                    break
                else:
                    step_size *= 0.5
                    
            if not improved:
                # If no improvement, reduce alpha and break or continue with smaller step
                alpha *= 0.5
                if alpha < 1e-4:
                    break
                    
        # Phase 2: High-precision polish
        if time.time() - start_time < time_limit - 5.0:
            alpha_polish = 0.005
            for step in range(500):
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                radii, centers, simplices = compute_facet_data(points)
                if len(radii) == 0:
                    break
                
                max_radius = np.max(radii)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                
                # Move only the worst facets
                if len(radii) > 5:
                    threshold = np.percentile(radii, 95)
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
                    # Move vertices towards the circumcenter
                    step_size = alpha_polish * radii[m]
                    
                    updates[i_idx] += step_size * (center - points[i_idx])
                    updates[j_idx] += step_size * (center - points[j_idx])
                    updates[k_idx] += step_size * (center - points[k_idx])
                    
                # Project to tangent
                for i in range(N):
                    proj = np.dot(updates[i], points[i])
                    updates[i] -= proj * points[i]
                
                points += updates
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
                alpha_polish *= 0.99

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
