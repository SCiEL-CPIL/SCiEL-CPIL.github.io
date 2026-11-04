import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _facet_data(points):
    """Compute circumradii, circumcenter directions, and simplices for the convex hull."""
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
    
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        radii = np.full(len(simplices), np.pi)
        centers = np.zeros((len(simplices), 3))
        centers[~valid] = [0, 0, 1]
        return radii, centers, simplices
    
    u = n / norms[:, np.newaxis]
    
    # Orient normals outward
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius: cos(r) = u · a (since all on unit sphere)
    cos_r = np.sum(u * a, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    return radii, u, simplices

def _smooth_objective_and_grad(points, tau):
    """
    Compute the log-sum-exp surrogate objective and its exact gradient.
    
    J = tau * log(sum_f exp(r_f / tau))
    
    Gradient of r_f with respect to point p_i (for i in facet f):
        dr_f/dp_i = -u_f / sin(r_f)
    
    So gradient of J:
        dJ/dp_i = sum_{f containing i} w_f * (-u_f / sin(r_f))
    
    where w_f = exp(r_f/tau) / sum_g exp(r_g/tau)
    
    Returns (J, grad) where grad has shape (N, 3).
    """
    radii, centers, simplices = _facet_data(points)
    if len(radii) == 0:
        return np.pi, np.zeros_like(points)
    
    # Soft-max weights
    r_scaled = radii / tau
    r_max = np.max(r_scaled)
    exp_terms = np.exp(r_scaled - r_max)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    J = tau * (r_max + np.log(sum_exp))
    
    # Gradient
    grad = np.zeros_like(points)
    sin_r = np.sin(radii)
    sin_r[sin_r < 1e-8] = 1e-8
    
    for m in range(len(radii)):
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        center = centers[m]
        w = weights[m]
        factor = w / sin_r[m]
        
        # dr_f/dp_i = -u_f / sin(r_f)
        # dJ/dp_i = w_f * dr_f/dp_i = -w_f * u_f / sin(r_f)
        # We want to descend, so we add -grad = +w_f * u_f / sin(r_f)
        # But here we return the gradient (for use in p -= lr * grad)
        grad[i_idx] -= factor * center
        grad[j_idx] -= factor * center
        grad[k_idx] -= factor * center
    
    return J, grad

def _fixed_topology_optimize(points, simplices, centers, radii, time_limit, max_steps=5000):
    """
    Optimize with fixed facet topology. This avoids re-computing the hull each step.
    Uses exact gradient of the smooth surrogate with annealing temperature.
    """
    start_time = time.time()
    n_facets = len(radii)
    
    # Adaptive learning rate and temperature
    tau = 0.15
    tau_min = 0.02
    lr = 0.08
    lr_min = 0.001
    patience = 0
    best_local = np.max(radii)
    best_points = points.copy()
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Anneal temperature
        tau = max(tau_min, tau * 0.9995)
        
        # Soft-max weights
        r_scaled = radii / tau
        r_max = np.max(r_scaled)
        exp_terms = np.exp(r_scaled - r_max)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        J = tau * (r_max + np.log(sum_exp))
        
        # Gradient
        sin_r = np.sin(radii)
        sin_r[sin_r < 1e-8] = 1e-8
        
        grad = np.zeros_like(points)
        for m in range(n_facets):
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex
            center = centers[m]
            w = weights[m]
            factor = w / sin_r[m]
            grad[i_idx] -= factor * center
            grad[j_idx] -= factor * center
            grad[k_idx] -= factor * center
        
        # Project to tangent plane
        for i in range(N):
            proj = np.dot(grad[i], points[i])
            grad[i] -= proj * points[i]
        
        # Adaptive step size: scale by typical radius
        step_size = lr * np.mean(radii)
        
        # Update
        new_points = points - step_size * grad
        new_points = _normalize(new_points)
        
        # Recompute facet data (radii and centers change as points move)
        a = new_points[simplices[:, 0]]
        b = new_points[simplices[:, 1]]
        c = new_points[simplices[:, 2]]
        
        n = np.cross(b - a, c - a)
        norms = np.linalg.norm(n, axis=1)
        valid = norms > 1e-10
        
        if not np.all(valid):
            # Topology broke; re-hull
            break
        
        u = n / norms[:, np.newaxis]
        centroids = (a + b + c) / 3.0
        dots = np.sum(u * centroids, axis=1)
        u[dots < 0] *= -1
        
        cos_r = np.sum(u * a, axis=1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        new_radii = np.arccos(cos_r)
        
        max_r = np.max(new_radii)
        
        if max_r < best_local - 1e-10:
            best_local = max_r
            best_points = new_points.copy()
            patience = 0
            lr = min(lr * 1.02, 0.15)
        else:
            patience += 1
            if patience > 50:
                lr *= 0.9
                patience = 0
                if lr < lr_min:
                    break
        
        points = new_points
        radii = new_radii
        centers = u
    
    return best_points

def _nelder_mead_polish(points, time_limit):
    """
    Nelder-Mead refinement on the worst facets.
    We parameterize each point as a 2D tangent vector around its current position.
    For simplicity, we do coordinate-wise perturbation on the worst facets.
    """
    start_time = time.time()
    best_points = points.copy()
    best_radius = np.max(_facet_data(points)[0])
    
    # Find worst facets
    radii, centers, simplices = _facet_data(points)
    if len(radii) == 0:
        return best_points
    
    # Focus on top 20% worst facets
    n_worst = max(1, len(radii) // 5)
    worst_idx = np.argsort(radii)[-n_worst:]
    worst_points = set()
    for m in worst_idx:
        for idx in simplices[m]:
            worst_points.add(idx)
    worst_points = list(worst_points)
    
    # Simple iterative refinement: perturb points in worst facets
    step_size = 0.05
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for p_idx in worst_points:
            if time.time() - start_time > time_limit:
                break
            
            best_r = np.max(_facet_data(points)[0])
            best_p = points[p_idx].copy()
            
            # Try small perturbations in random tangent directions
            for _ in range(8):
                # Random tangent vector
                v = np.random.randn(3)
                v -= np.dot(v, points[p_idx]) * points[p_idx]
                v /= (np.linalg.norm(v) + 1e-12)
                
                new_p = points[p_idx] + step_size * v
                new_p = new_p / np.linalg.norm(new_p)
                
                trial = points.copy()
                trial[p_idx] = new_p
                trial = _normalize(trial)
                
                r_trial = np.max(_facet_data(trial)[0])
                if r_trial < best_r - 1e-12:
                    best_r = r_trial
                    best_p = new_p
                    improved = True
            
            points[p_idx] = best_p
        
        points = _normalize(points)
        current_r = np.max(_facet_data(points)[0])
        if current_r < best_radius - 1e-12:
            best_radius = current_r
            best_points = points.copy()
        
        step_size *= 0.95
        if step_size < 1e-5:
            break
    
    return best_points

def greedy_init(n_points, seed=0):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        candidates = np.random.randn(2000, 3)
        candidates = _normalize(candidates)
        
        existing = np.array(points)
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
    
    return np.array(points)

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
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
    
    # 2. Greedy initialization
    init_greedy = greedy_init(N, seed=42)
    initializations.append(init_greedy)
    
    # 3. Another greedy
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(init_greedy2)
    
    # 4. Fibonacci + small noise (multiple seeds)
    for s in range(5):
        np.random.seed(100 + s * 17)
        init = init_fib + np.random.randn(N, 3) * (0.05 + 0.05 * s / 5)
        init = _normalize(init)
        initializations.append(init)
    
    # 5. Pure random
    np.random.seed(999)
    init_rand = np.random.randn(N, 3)
    init_rand = _normalize(init_rand)
    initializations.append(init_rand)
    
    # 6. More diverse random restarts
    for s in range(10):
        np.random.seed(5000 + s * 31)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.15 * s / 10)
        init = _normalize(init)
        initializations.append(init)
    
    max_restarts = len(initializations)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        points = initializations[restart].copy()
        
        # Phase 1: Smooth gradient descent with annealing (re-hull each step)
        tau = 0.2
        alpha = 0.06
        no_improve = 0
        best_local = np.pi
        
        for step in range(2000):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            radii, centers, simplices = _facet_data(points)
            if len(radii) == 0:
                break
            
            max_radius = np.max(radii)
            
            if max_radius < best_local - 1e-10:
                best_local = max_radius
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    alpha *= 0.95
                    if alpha < 0.005:
                        break
                if no_improve > 100:
                    break
            
            # Anneal
            tau = max(0.03, tau * 0.999)
            
            J, grad = _smooth_objective_and_grad(points, tau)
            
            # Project to tangent
            for i in range(N):
                proj = np.dot(grad[i], points[i])
                grad[i] -= proj * points[i]
            
            points = points - alpha * grad
            points = _normalize(points)
        
        # Check improvement
        radii, centers, simplices = _facet_data(points)
        if len(radii) > 0:
            max_radius = np.max(radii)
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
        
        # Phase 2: Fixed-topology optimization (faster, no re-hull)
        if time.time() - start_time < time_limit - 15.0:
            radii, centers, simplices = _facet_data(points)
            if len(radii) > 0:
                time_for_phase2 = min(20.0, time_limit - 15.0 - (time.time() - start_time))
                polished = _fixed_topology_optimize(points, simplices, centers, radii, time_for_phase2, max_steps=3000)
                
                r_polished = np.max(_facet_data(polished)[0])
                if r_polished < best_radius:
                    best_radius = r_polished
                    best_points = polished.copy()
        
        # Phase 3: Nelder-Mead polish on worst facets
        if time.time() - start_time < time_limit - 5.0:
            if best_points is not None:
                time_for_nm = min(10.0, time_limit - 5.0 - (time.time() - start_time))
                if time_for_nm > 2.0:
                    np.random.seed(restart * 7 + 1)
                    nm_result = _nelder_mead_polish(best_points, time_for_nm)
                    r_nm = np.max(_facet_data(nm_result)[0])
                    if r_nm < best_radius:
                        best_radius = r_nm
                        best_points = nm_result.copy()
    
    if best_points is None:
        best_points = init_fib
    
    return best_points

