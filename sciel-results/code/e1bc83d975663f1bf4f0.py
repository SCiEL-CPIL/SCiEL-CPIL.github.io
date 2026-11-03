import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_circumradii(pts, simplices):
    """Vectorized computation of circumradii for all facets.
    Returns array of radii (radians) and array of circumcenter unit vectors."""
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    n = np.cross(a - b, a - c)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm > 1e-10
    safe_norm = np.maximum(n_norm, 1e-10)
    n = n / safe_norm[:, None]
    centroids = (a + b + c) / 3.0
    signs = np.where(np.einsum('ij,ij->i', n, centroids) < 0, -1.0, 1.0)
    n = n * signs[:, None]
    dots = np.clip(np.einsum('ij,ij->i', n, a), -1.0, 1.0)
    radii = np.arccos(dots)
    radii[~mask] = 0.0
    return radii, n, mask

def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.degrees(np.max(radii)))
    except Exception:
        return 180.0

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, n_iters=300, lr=0.005):
    """Repulsion-based refinement: each point is pushed away by all others via Coulomb-like force."""
    pts = pts.copy()
    for it in range(n_iters):
        D = pts @ pts.T
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        # Project to tangent plane
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        forces /= norms
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _smooth_surrogate(pts, beta=10.0):
    """Smooth surrogate: log-sum-exp of squared facet radii (soft-max)."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    radii, cc, mask = _facet_circumradii(pts, hull.simplices)
    if not np.any(mask):
        return None
    valid = mask
    r_valid = radii[valid]
    cc_valid = cc[valid]
    simp_valid = hull.simplices[valid]
    r2 = r_valid * r_valid
    m = np.max(r2)
    exps = np.exp((r2 - m) * beta)
    s = np.sum(exps)
    if s < 1e-300:
        weights = np.ones_like(exps) / len(exps)
    else:
        weights = exps / s
    surrogate = m + np.log(s) / beta
    return surrogate, weights, r_valid, cc_valid, simp_valid

def _surrogate_gradient(pts, weights, simplices, circumcenters):
    """Gradient of smooth surrogate w.r.t. each point (in R^3, then projected to tangent)."""
    grad = np.zeros_like(pts)
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-8:
            continue
        c = circumcenters[k]
        a = pts[simplices[k, 0]]
        cos_r = np.clip(np.dot(c, a), -1.0, 1.0)
        sin_r = np.sqrt(max(1.0 - cos_r * cos_r, 1e-12))
        r_k = np.arccos(cos_r)
        for j in range(3):
            idx = simplices[k, j]
            v = pts[idx]
            d = c - v
            d_t = d - np.dot(d, v) * v
            d_t_norm = np.linalg.norm(d_t)
            if d_t_norm < 1e-10:
                continue
            d_t = d_t / d_t_norm
            mag = 2.0 * r_k / sin_r * w
            grad[idx] -= mag * d_t
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return grad

def _lloyd_step(pts, step_size=0.05):
    """
    Greedy Lloyd step: Find the facet with the largest circumradius.
    Move its 3 vertices towards the circumcenter by step_size.
    Returns updated points and the index of the worst facet (or -1 if none).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts, -1
        
    radii, cc, mask = _facet_circumradii(pts, hull.simplices)
    if not np.any(mask):
        return pts, -1
        
    # Find the worst facet
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return pts, -1
        
    # Get the index of the max radius among valid ones
    valid_radii = radii[valid_indices]
    worst_local_idx = np.argmax(valid_radii)
    worst_global_idx = valid_indices[worst_local_idx]
    
    # Get the facet details
    simplex = hull.simplices[worst_global_idx]
    circumcenter = cc[worst_global_idx]
    r_worst = radii[worst_global_idx]
    
    # Only move if the radius is significant (avoid noise)
    if r_worst < 1e-4:
        return pts, -1
        
    new_pts = pts.copy()
    
    # Move vertices towards circumcenter
    # The direction is from vertex to circumcenter, projected to tangent
    for idx in simplex:
        v = new_pts[idx]
        # Vector from v to c
        d = circumcenter - v
        # Project to tangent plane at v
        d_t = d - np.dot(d, v) * v
        d_t_norm = np.linalg.norm(d_t)
        if d_t_norm < 1e-10:
            continue
        d_t = d_t / d_t_norm
        # Move
        new_pts[idx] = v + step_size * d_t
        
    new_pts = _normalize(new_pts)
    return new_pts, worst_global_idx

def _local_optimize_hybrid(pts, deadline, initial_step=0.03):
    """
    Hybrid local optimization:
    1. Greedy Lloyd steps to quickly reduce the max radius.
    2. Smooth surrogate gradient steps to refine and prevent local minima.
    """
    best_pts = pts.copy()
    best_rad = _covering_radius_deg(pts)
    step_size = initial_step
    stagnant_count = 0
    
    while time.time() < deadline:
        # 1. Try a few Lloyd steps
        improved = False
        for _ in range(5):
            if time.time() > deadline:
                break
            new_pts, _ = _lloyd_step(best_pts, step_size)
            new_rad = _covering_radius_deg(new_pts)
            if new_rad < best_rad - 1e-6:
                best_rad = new_rad
                best_pts = new_pts
                improved = True
                stagnant_count = 0
            else:
                # If Lloyd doesn't improve, shrink step size slightly
                step_size *= 0.9
        
        # 2. Try Smooth Surrogate Gradient Step
        if time.time() > deadline:
            break
        
        surr = _smooth_surrogate(best_pts, beta=10.0)
        if surr is None:
            break
            
        surrogate_val, weights, r_valid, cc_valid, simp_valid = surr
        grad = _surrogate_gradient(best_pts, weights, simp_valid, cc_valid)
        
        # Normalize gradient to unit length for consistent step size
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm = np.where(grad_norm < 1e-12, 1.0, grad_norm)
        grad = grad / grad_norm
        
        # Step
        new_pts = best_pts + step_size * grad
        new_pts = _normalize(new_pts)
        
        new_rad = _covering_radius_deg(new_pts)
        if new_rad < best_rad - 1e-6:
            best_rad = new_rad
            best_pts = new_pts
            stagnant_count = 0
        else:
            stagnant_count += 1
            step_size *= 0.95
            if step_size < 1e-5:
                break
                
    return best_pts, best_rad

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin

    # Initialize with multiple strategies
    best_points = _fibonacci_spiral(0.0)
    best_rad = _covering_radius_deg(best_points)
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_spiral(0.0))
    inits.append(_fibonacci_spiral(0.3))
    inits.append(_fibonacci_spiral(0.7))
    
    rng = np.random.RandomState(42)
    for _ in range(2):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
    
    # Electrostatic-refined from Fibonacci
    inits.append(_electrostatic_refine(_fibonacci_spiral(0.0), n_iters=300, lr=0.005))
    
    # Evaluate all initializations
    for init in inits:
        if time.time() - start_time > max_time - 20.0:
            break
        rad = _covering_radius_deg(init)
        if rad < best_rad:
            best_rad = rad
            best_points = init.copy()
            
    # Sort by quality and take top 3
    init_list = []
    for init in inits:
        rad = _covering_radius_deg(init)
        init_list.append((rad, init.copy()))
    init_list.sort(key=lambda x: x[0])
    top_inits = [x[1] for x in init_list[:3]]
    
    # Local optimization on top initializations using Hybrid Method
    for init_pts in top_inits:
        if time.time() - start_time > max_time - 15.0:
            break
        
        pts = init_pts.copy()
        cur_rad = _covering_radius_deg(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()
        
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time < 10.0:
            break
        
        opt_pts, opt_rad = _local_optimize_hybrid(pts, start_time + remaining_time - 5.0, initial_step=0.03)
        
        if opt_rad < best_rad:
            best_rad = opt_rad
            best_points = opt_pts.copy()
            
    # Final polish: very fine local optimization
    if time.time() - start_time < max_time - 10.0:
        polish_pts, polish_rad = _local_optimize_hybrid(best_points, max_time - 3.0, initial_step=0.005)
        if polish_rad < best_rad:
            best_rad = polish_rad
            best_points = polish_pts.copy()
            
    return best_points
