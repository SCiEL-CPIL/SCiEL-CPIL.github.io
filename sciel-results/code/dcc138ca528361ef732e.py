import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_circumradii(points, hull):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of circumradii (angular)
        centers: (M, 3) array of circumcenters on sphere
        facets: (M, 3) array of facet indices
    """
    facets = hull.simplices
    M = len(facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = np.where(dist > np.pi / 2, np.pi - dist, dist)
    
    return radii, centers, facets

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        radii, centers, facets = _compute_circumradii(points, hull)
        return float(np.max(radii)) * 180.0 / np.pi
    except Exception:
        return 180.0

def _normalize_points(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _soft_max_objective_and_grad(points, beta=50.0):
    """
    Compute the soft-max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to point positions.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.0, np.zeros_like(points)
    
    radii, centers, facets = _compute_circumradii(points, hull)
    M = len(radii)
    
    if M == 0:
        return 0.0, np.zeros_like(points)
    
    r_max = np.max(radii)
    if r_max < 1e-15:
        return 0.0, np.zeros_like(points)
    
    shifted = beta * (radii - r_max)
    log_sum_exp = r_max + np.log(np.sum(np.exp(shifted))) / beta
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(points)
    
    for k in range(M):
        if weights[k] < 1e-10:
            continue
        
        tri = facets[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        c = centers[k]
        r_k = radii[k]
        
        if r_k < 1e-12:
            continue
        
        dot_cp1 = np.dot(c, p1)
        sin_r1 = np.sqrt(max(0.0, 1.0 - dot_cp1 * dot_cp1))
        
        if sin_r1 > 1e-12:
            grad_p1 = -c / sin_r1
            grad_p1 -= np.dot(grad_p1, p1) * p1
            grad[tri[0]] += weights[k] * grad_p1
        
        dot_cp2 = np.dot(c, p2)
        sin_r2 = np.sqrt(max(0.0, 1.0 - dot_cp2 * dot_cp2))
        if sin_r2 > 1e-12:
            grad_p2 = -c / sin_r2
            grad_p2 -= np.dot(grad_p2, p2) * p2
            grad[tri[1]] += weights[k] * grad_p2
        
        dot_cp3 = np.dot(c, p3)
        sin_r3 = np.sqrt(max(0.0, 1.0 - dot_cp3 * dot_cp3))
        if sin_r3 > 1e-12:
            grad_p3 = -c / sin_r3
            grad_p3 -= np.dot(grad_p3, p3) * p3
            grad[tri[2]] += weights[k] * grad_p3
    
    return float(log_sum_exp), grad

def _energy_gradient(points):
    """
    Compute the tangent gradient of the potential V = sum_{i<j} x_i . x_j.
    Minimizing V spreads points apart (Bures-Wasserstein like behavior).
    grad_k V = sum_{j != k} x_j = sum_all - x_k.
    """
    S = np.sum(points, axis=0)
    grads = S[None, :] - points
    dots = np.sum(points * grads, axis=1, keepdims=True)
    grads_tangent = grads - dots * points
    return grads_tangent

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize_points(points)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points = _normalize_points(points)
    
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Energy Gradient Descent (Smooth, Global)
    # This uses the Bures-like energy to spread points out efficiently.
    energy_step = 0.01
    energy_deadline = start_time + 30.0 # Run for 30s
    
    while time.time() < energy_deadline:
        grads = _energy_gradient(points)
        # Normalize gradients to ensure stable steps
        norms = np.linalg.norm(grads, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        grads_norm = grads / norms
        
        new_points = points - energy_step * grads_norm
        new_points = _normalize_points(new_points)
        
        # Check for improvement in covering radius occasionally
        # Since energy minimization is a proxy, we just track best radius
        # but update points always as energy decreases
        points = new_points
        
        # Adaptive step size
        # If energy stops decreasing significantly, we might be stuck, 
        # but for now we just run until time limit as it's smooth.
        # We can add a check if radius improves.
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Decay step size slowly
        energy_step *= 0.999

    # Phase 2: Local Refinement (Lloyd + SoftMax)
    points = best_points.copy()
    step_size = 0.03
    min_step_size = 1e-7
    no_improve_count = 0
    max_no_improve = 40
    use_gradient = False
    gradient_step = 0.02
    beta = 30.0
    
    while time.time() < deadline - 25.0:
        try:
            hull = ConvexHull(points)
            radii, centers, facets = _compute_circumradii(points, hull)
        except Exception:
            kick = rng.normal(0, 0.02, points.shape)
            points += kick
            points = _normalize_points(points)
            continue
        
        M = len(facets)
        if M == 0:
            break
        
        # For each point, find the adjacent triangle with the maximum radius
        point_max_radius = np.zeros(N)
        point_max_center = np.zeros((N, 3))
        
        for k in range(M):
            r_k = radii[k]
            if r_k <= 0:
                continue
            c_k = centers[k]
            tri = facets[k]
            for idx in tri:
                if r_k > point_max_radius[idx]:
                    point_max_radius[idx] = r_k
                    point_max_center[idx] = c_k
        
        if use_gradient:
            # Use soft-max gradient descent
            obj, grad = _soft_max_objective_and_grad(points, beta)
            
            grad_tangent = np.zeros_like(points)
            for i in range(N):
                g = grad[i]
                g_tan = g - np.dot(g, points[i]) * points[i]
                norm_g = np.linalg.norm(g_tan)
                if norm_g > 1e-12:
                    grad_tangent[i] = g_tan / norm_g
            
            new_points = points - gradient_step * grad_tangent
            new_points = _normalize_points(new_points)
            
            current_radius = _compute_covering_radius(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
                no_improve_count = 0
                gradient_step = min(gradient_step * 1.05, 0.05)
                beta = min(beta * 1.03, 150.0)
            else:
                gradient_step = max(gradient_step * 0.85, 1e-6)
                no_improve_count += 1
        else:
            # Lloyd-style: move points towards their worst facet center
            max_move = np.zeros_like(points)
            
            for i in range(N):
                if point_max_radius[i] > 0:
                    v = point_max_center[i] - points[i]
                    dot_vp = np.dot(v, points[i])
                    tangent_v = v - dot_vp * points[i]
                    norm_tv = np.linalg.norm(tangent_v)
                    if norm_tv > 1e-12:
                        # Scale by radius to prioritize large holes
                        max_move[i] = (tangent_v / norm_tv) * point_max_radius[i]
            
            new_points = points + step_size * max_move
            new_points = _normalize_points(new_points)
            
            current_radius = _compute_covering_radius(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
                no_improve_count = 0
                step_size = min(step_size * 1.04, 0.08)
            else:
                step_size = max(step_size * 0.82, min_step_size)
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Stagnation: switch strategy or random kick
            if not use_gradient:
                use_gradient = True
                gradient_step = 0.02
                beta = 30.0
                step_size = 0.03
            else:
                # Random kick
                kick = rng.normal(0, 0.04, points.shape)
                points += kick
                points = _normalize_points(points)
                step_size = 0.02
                gradient_step = 0.01
                beta = 20.0
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
            no_improve_count = 0
    
    # Phase 3: High-precision polish using exact max facet targeting
    points = best_points.copy()
    polish_deadline = deadline - 5.0
    polish_step = 0.003
    
    while time.time() < polish_deadline:
        try:
            hull = ConvexHull(points)
            radii, centers, facets = _compute_circumradii(points, hull)
        except Exception:
            break
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        worst_facet = facets[max_idx]
        worst_center = centers[max_idx]
        
        # Move points in the worst facet towards the circumcenter
        for idx in worst_facet:
            v = worst_center - points[idx]
            dot_vp = np.dot(v, points[idx])
            tangent_v = v - dot_vp * points[idx]
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-12:
                points[idx] += polish_step * tangent_v / norm_tv
        
        points = _normalize_points(points)
        
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            # Try smaller step
            polish_step *= 0.6
            if polish_step < 1e-7:
                break
            # Restore and retry
            points = best_points.copy()
    
    return best_points
