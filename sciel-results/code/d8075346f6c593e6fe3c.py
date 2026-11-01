import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_hull_data(pts):
    """
    Compute convex hull and extract facet data.
    Returns:
        radii: (F,) array of covering radii for each facet
        normals: (F, 3) array of outward normals for each facet
        faces: (F, 3) array of vertex indices for each facet
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.array([np.pi]), np.array([[0,0,0]]), faces[:1]

        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward (away from origin)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Calculate covering radius for each facet
        # The furthest point on the sphere from the facet plane is the pole.
        # Distance from origin to plane is |n . A| = |n . B| = |n . C|
        # The angular radius is arccos(n . A)
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces[valid]
    except Exception:
        return np.array([np.pi]), np.array([[0,0,0]]), np.array([[-1, -1, -1]])

def _covering_radius(pts):
    """Compute the exact covering radius."""
    radii, _, _ = _get_hull_data(pts)
    return np.max(radii)

def _soft_hull_objective_and_grad(pts, t=0.05):
    """
    Compute a smooth surrogate objective (Log-Sum-Exp of radii) 
    and its gradient on the sphere.
    """
    radii, normals, faces = _get_hull_data(pts)
    if len(radii) == 0:
        return 0.0, np.zeros((N, 3))
    
    # LSE objective: t * log(sum(exp(r_i / t)))
    # We want to minimize the max radius, so we maximize the negative LSE.
    # For gradient descent, we minimize the LSE.
    
    # Numerical stability
    max_r = np.max(radii)
    scaled = radii - max_r
    weights = np.exp(scaled / t)
    weights /= np.sum(weights)
    
    obj = max_r + t * np.log(np.sum(weights))
    
    # Gradient of LSE wrt points
    # d LSE / d p_k = sum_i w_i * (d r_i / d p_k)
    # r_i = arccos(n_i . p_a) for p_a in facet i
    # dr_i / d p_a = -n_i / sqrt(1 - (n_i . p_a)^2)
    
    grad = np.zeros((N, 3))
    
    # For each facet, add contribution to its 3 vertices
    # r_i depends on n_i and p_a.
    # n_i is normalized cross product of edges.
    # This is complex. A simpler approximation for gradient:
    # The pole direction is n_i. The vertex p_a should move towards n_i to reduce r_i.
    # The sensitivity is roughly 1 / sin(r_i).
    
    sin_r = np.sin(radii)
    sin_r = np.clip(sin_r, _EPS, None)
    
    # Gradient contribution for each facet i to its vertices
    # We want to decrease r_i. 
    # dr_i/dp_a approx -n_i / sin(r_i)
    # So gradient of objective wrt p_a is sum_i w_i * (-n_i / sin(r_i)) * (if a in facet i)
    
    # Actually, to minimize r, we want to move p in direction -grad r.
    # grad r wrt p is roughly n / sin(r).
    # So we move p in direction -n / sin(r).
    
    # Let's accumulate the "force" pushing vertices towards the pole.
    # Force on vertex a from facet i is w_i * n_i / sin(r_i)
    # We subtract this from p to minimize r.
    
    forces = np.zeros((N, 3))
    
    for i in range(len(radii)):
        w = weights[i]
        n = normals[i]
        s = 1.0 / sin_r[i]
        f = w * s * n
        v_indices = faces[i]
        # Distribute force equally or just add to each?
        # Each vertex in the facet contributes to the radius.
        # Moving any of them reduces the radius.
        # The derivative is with respect to each coordinate.
        # Let's just add the force vector to each of the 3 vertices.
        # The normalization on the sphere handles the rest.
        forces[v_indices[0]] += f
        forces[v_indices[1]] += f
        forces[v_indices[2]] += f
        
    # Gradient of objective is forces.
    # We want to move in direction -gradient.
    # So update is p - lr * forces.
    
    return obj, forces

def _riemannian_grad_descent(pts, deadline, rng, t=0.05, lr=0.01):
    """
    Perform Riemannian Gradient Descent on the LSE surrogate.
    """
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_r = _covering_radius(current_pts)
    
    # Adaptive learning rate
    current_lr = lr
    
    while time.time() < deadline:
        obj, grad = _soft_hull_objective_and_grad(current_pts, t)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < _EPS:
            break
        grad /= grad_norm
        
        # Move points: p_new = p - lr * grad
        new_pts = current_pts - current_lr * grad
        
        # Normalize to sphere
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Check improvement using exact hull radius
        new_r = _covering_radius(new_pts)
        
        if new_r < best_r - 1e-9:
            best_r = new_r
            best_pts = new_pts.copy()
            current_pts = new_pts
            # Increase LR slightly if making progress
            current_lr = min(current_lr * 1.05, 0.05)
        else:
            # Decrease LR if stuck
            current_lr *= 0.8
            if current_lr < 1e-6:
                # Random perturbation to escape local minima
                current_pts = best_pts + rng.standard_normal((N, 3)) * 0.05
                current_pts /= np.linalg.norm(current_pts, axis=1, keepdims=True)
                current_lr = 0.02
                t = max(t * 0.8, 0.01) # Anneal the LSE temperature

    return best_pts, best_r

def _polish_with_hull(pts, deadline):
    """
    Final local search using exact hull radius.
    """
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_r = _covering_radius(current_pts)
    
    step_size = 0.01
    
    while time.time() < deadline:
        radii, normals, faces = _get_hull_data(current_pts)
        if len(radii) == 0:
            break
        
        worst_idx = np.argmax(radii)
        pole = normals[worst_idx]
        v_indices = faces[worst_idx]
        
        improved = False
        for v_idx in v_indices:
            dir_vec = pole - current_pts[v_idx]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < _EPS:
                continue
            dir_vec /= dir_norm
            
            new_pts = current_pts.copy()
            new_pts[v_idx] += step_size * dir_vec
            new_pts[v_idx] /= (np.linalg.norm(new_pts[v_idx]) + _EPS)
            
            new_r = _covering_radius(new_pts)
            if new_r < best_r - 1e-9:
                best_r = new_r
                best_pts = new_pts.copy()
                current_pts = new_pts
                improved = True
                break
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-6:
                break

    return best_pts, best_r

def construct():
    # Safety margin: Evaluator kills at 150s. We stop at 140s to return.
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    # Initial configuration
    pts = _initial_fib()
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    
    # Phase 1: Riemannian Gradient Descent on LSE surrogate
    # This is the structural change: smooth optimization instead of random SA.
    time_limit_1 = deadline - 60.0 # Leave 60s for polish and restarts
    gd_pts, gd_r = _riemannian_grad_descent(pts, time_limit_1, rng)
    
    if gd_r < best_r:
        best_r = gd_r
        best_pts = gd_pts
    
    # Phase 2: Restart with perturbation if time allows
    if time.time() < deadline - 30.0:
        restart_pts = best_pts + rng.standard_normal((N, 3)) * 0.1
        restart_pts /= np.linalg.norm(restart_pts, axis=1, keepdims=True)
        time_limit_2 = deadline - 30.0
        gd_pts2, gd_r2 = _riemannian_grad_descent(restart_pts, time_limit_2, rng)
        if gd_r2 < best_r:
            best_r = gd_r2
            best_pts = gd_pts2

    # Phase 3: Final Polish with exact hull
    if time.time() < deadline - 5.0:
        polish_pts, polish_r = _polish_with_hull(best_pts, deadline - 2.0)
        if polish_r < best_r:
            best_r = polish_r
            best_pts = polish_pts

    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts /= (norms + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
