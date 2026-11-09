"""Spherical Riemannian gradient flow for minimizing covering radius of 50 points."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _spherical_circumcenter(tri):
    """
    Compute the spherical circumcenter of a triangle on the unit sphere.
    Returns the circumcenter (unit vector) and the circumradius (angular distance).
    """
    a, b, c = tri[0], tri[1], tri[2]
    
    u = a - b
    v = a - c
    w = np.cross(u, v)
    norm_w = np.linalg.norm(w)
    
    if norm_w < 1e-12:
        # Degenerate case
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_ac = np.arccos(np.clip(np.dot(a, c), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        
        if d_ab >= d_ac and d_ab >= d_bc:
            mid = a + b
        elif d_ac >= d_ab and d_ac >= d_bc:
            mid = a + c
        else:
            mid = b + c
            
        norm_mid = np.linalg.norm(mid)
        if norm_mid < 1e-12:
            return a, np.pi/2
        p = mid / norm_mid
        r = np.arccos(np.clip(np.dot(p, a), -1, 1))
        return p, r

    p = w / norm_w
    
    # Choose the correct sign: circumcenter should be on same side as centroid
    centroid = a + b + c
    norm_centroid = np.linalg.norm(centroid)
    if norm_centroid > 1e-12:
        centroid_norm = centroid / norm_centroid
        if np.dot(p, centroid_norm) < 0:
            p = -p
            
    d1 = np.arccos(np.clip(np.dot(p, a), -1, 1))
    d2 = np.arccos(np.clip(np.dot(p, b), -1, 1))
    d3 = np.arccos(np.clip(np.dot(p, c), -1, 1))
    r = np.mean([d1, d2, d3])
    
    return p, r

def _compute_covering_radius(pts):
    """
    Compute the covering radius by finding the convex hull and checking each facet's circumradius.
    Returns the max radius and indices of active facets.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.array([], dtype=int)
    
    radii = []
    for simplex in hull.simplices:
        tri = pts[simplex]
        cc, r = _spherical_circumcenter(tri)
        radii.append(r)
        
    radii = np.array(radii)
    max_radius = np.max(radii)
    active_indices = np.where(radii > max_radius - 1e-6)[0]
    
    return max_radius, active_indices

def _gradient_from_facets(pts, active_indices, hull):
    """
    Compute the Riemannian gradient for the max circumradius objective.
    For each active facet, we push its vertices toward the circumcenter.
    """
    grad = np.zeros_like(pts)
    
    for i in active_indices:
        simplex = hull.simplices[i]
        tri = pts[simplex]
        cc, r = _spherical_circumcenter(tri)
        
        if r < 1e-10:
            continue
            
        for j, idx in enumerate(simplex):
            v = pts[idx]
            dot_vc = np.dot(v, cc)
            # Tangent vector from v to cc
            t = cc - dot_vc * v
            norm_t = np.linalg.norm(t)
            
            if norm_t > 1e-12:
                u = t / norm_t
                grad[idx] += u
    
    return grad

def _riemannian_step(pts, grad, step_size):
    """
    Apply a Riemannian step: move each point along its gradient direction on the sphere.
    Uses the exponential map: p_new = cos(step) * p + sin(step) * (grad/|grad|) * |grad|
    Simplified: p_new = normalize(p + step * grad)
    """
    new_pts = np.zeros_like(pts)
    for i in range(N):
        g = grad[i]
        norm_g = np.linalg.norm(g)
        if norm_g > 1e-14:
            # Riemannian exponential map
            angle = step_size * norm_g
            if angle < 1e-14:
                new_pts[i] = pts[i]
            else:
                g_unit = g / norm_g
                new_pts[i] = np.cos(angle) * pts[i] + np.sin(angle) * g_unit
        else:
            new_pts[i] = pts[i]
    return _normalize(new_pts)

def _optimize_single_start(pts, max_iters=3000, timeout=90.0):
    """
    Optimize a single starting configuration using Riemannian gradient descent
    with adaptive step size and perturbation-based escape from local minima.
    """
    start_time = time.time()
    current_radius, active_indices = _compute_covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    step_size = 0.03
    min_step_size = 1e-7
    patience = 50
    patience_counter = 0
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Recompute active indices
        radii = []
        for simplex in hull.simplices:
            tri = pts[simplex]
            cc, r = _spherical_circumcenter(tri)
            radii.append(r)
        radii = np.array(radii)
        max_radius = np.max(radii)
        active_indices = np.where(radii > max_radius - 1e-6)[0]
        
        if max_radius < best_radius - 1e-9:
            best_radius = max_radius
            best_pts = pts.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if len(active_indices) == 0:
            break
            
        if patience_counter > patience:
            # Perturb to escape local minimum
            pts = pts + 0.02 * np.random.randn(N, 3)
            pts = _normalize(pts)
            step_size = 0.03
            patience_counter = 0
            continue
        
        # Compute gradient
        grad = _gradient_from_facets(pts, active_indices, hull)
        
        # Check if gradient is zero
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            # Stuck, perturb
            pts = pts + 0.01 * np.random.randn(N, 3)
            pts = _normalize(pts)
            step_size = 0.03
            continue
        
        # Try the step
        new_pts = _riemannian_step(pts, grad, step_size)
        new_radius, _ = _compute_covering_radius(new_pts)
        
        if new_radius < max_radius - 1e-12:
            pts = new_pts
            patience_counter = 0
        else:
            # Backtrack: reduce step size
            step_size *= 0.5
            if step_size < min_step_size:
                # Perturb
                pts = pts + 0.01 * np.random.randn(N, 3)
                pts = _normalize(pts)
                step_size = 0.03
    
    return best_pts, best_radius

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _random_rotation():
    """Generate a random 3x3 rotation matrix."""
    M = np.random.randn(3, 3)
    Q, R = np.linalg.qr(M)
    d = np.sign(np.linalg.det(Q))
    if d < 0:
        Q[:, 0] *= -1
    return Q

def construct():
    """
    Main function: generate initial configurations and optimize using
    Riemannian gradient flow. Returns the best configuration found.
    """
    best_pts = None
    best_radius = np.pi
    
    start_time = time.time()
    time_budget = 110.0
    
    np.random.seed(42)
    
    # Try multiple diverse starting configurations
    starts = []
    
    # Fibonacci spiral with various random rotations
    base_pts = _fibonacci_spiral()
    for k in range(4):
        Q = _random_rotation()
        starts.append(base_pts @ Q.T)
    
    # Random uniform starts
    for k in range(2):
        pts = np.random.randn(N, 3)
        starts.append(_normalize(pts))
    
    # One more rotated Fibonacci
    Q = _random_rotation()
    starts.append(base_pts @ Q.T)
    
    time_per_start = time_budget / len(starts)
    
    for k, start_pts in enumerate(starts):
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        timeout = min(time_per_start, remaining - 2.0)
        if timeout < 5.0:
            break
            
        current_pts, current_radius = _optimize_single_start(
            start_pts.copy(), max_iters=2000, timeout=timeout
        )
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts
            
        # Check time
        if time.time() - start_time > time_budget:
            break
    
    if best_pts is None:
        best_pts = _fibonacci_spiral()
    
    return best_pts

# EVOLVE-BLOCK-END
