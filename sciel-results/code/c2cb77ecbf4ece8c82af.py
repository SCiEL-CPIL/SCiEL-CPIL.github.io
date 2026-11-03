"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: grid
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of spherical triangle a,b,c
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / norm_v
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
            
        if np.dot(u, n) > 0:
            circum_center = u
        else:
            circum_center = -u
            
        d_a = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(circum_center, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(circum_center, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        
        max_cr = max(max_cr, cr)
        
    return max_cr

def _covering_radius_grid(pts, n_grid=100):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _spherical_lloyd(pts, max_iter=1000, tol=1e-8):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point to the centroid of its Voronoi cell.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        incident_facets = [[] for _ in range(n)]
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            normal = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            if np.dot(normal, a) < 0:
                normal = -normal
                
            for idx in simplex:
                incident_facets[idx].append(normal)
                
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        
        for i in range(n):
            if len(incident_facets[i]) == 0:
                new_pts[i] = pts[i]
                continue
                
            centroid = np.sum(incident_facets[i], axis=0)
            norm_c = np.linalg.norm(centroid)
            
            if norm_c < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm_c
                
            move = np.linalg.norm(new_pts[i] - pts[i])
            max_move = max(max_move, move)
            
        pts = _normalize(new_pts)
        
        if max_move < tol:
            break
            
    return pts

def _optimize_facet_shrink(pts, max_iter=1000, initial_step=0.5, decay=0.995, beta_start=2.0, beta_end=20.0):
    """
    Optimizes the configuration by moving each point towards the weighted
    average of the circumcenters of its incident facets.
    Uses adaptive step size and increasing beta to focus on worst facets.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))
        
        max_cr_global = 0.0
        facet_data = []

        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            
            n_face = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n_face)
            if norm_n < 1e-12:
                continue
            n_face = n_face / norm_n
            if np.dot(n_face, a) < 0:
                n_face = -n_face
                
            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u
                
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            cr = max(d_a, d_b, d_c)
            
            max_cr_global = max(max_cr_global, cr)
            facet_data.append((simplex, cc, cr))
            
        if max_cr_global < 1e-6:
            break

        # Interpolate beta from beta_start to beta_end
        progress = it / max_iter
        beta = beta_start + (beta_end - beta_start) * progress
        
        for simplex, cc, cr in facet_data:
            w = math.exp(beta * (cr - max_cr_global))
            
            for idx in simplex:
                weights_sum[idx] += w
                vec_sum[idx] += w * cc
                
        moved = False
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vec = np.linalg.norm(vec_tan)
                    
                    if norm_vec > 1e-10:
                        pts[i] = pts[i] + step_size * vec_tan
                        moved = True
                        
        if not moved:
            break
            
        pts = _normalize(pts)
        
        # Decay step size
        step_size *= decay
        
        if step_size < 1e-8:
            break

    return pts

def _soft_max_covering_radius(pts, beta=50.0):
    """
    Compute soft-max covering radius and its gradient w.r.t. point positions.
    Uses the convex hull to find facets and their circumradii.
    Returns (soft_max_cr, gradient_array) where gradient has shape (n, 3).
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: return a large value and zero gradient
        return 1.0, np.zeros((n, 3))
    
    simplices = hull.simplices
    n_facets = len(simplices)
    
    if n_facets == 0:
        return 1.0, np.zeros((n, 3))
    
    # Vectorized circumcenter computation
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # v = cross(a-b, b-c)
    v = np.cross(a - b, b - c)
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    valid = norm_v.squeeze() > 1e-12
    norm_v[~valid] = 1.0
    u = v / norm_v
    
    # n_face = cross(b-a, c-a)
    n_face = np.cross(b - a, c - a)
    norm_n = np.linalg.norm(n_face, axis=1, keepdims=True)
    valid2 = norm_n.squeeze() > 1e-12
    norm_n[~valid2] = 1.0
    n_face = n_face / norm_n
    
    # Orient normals outward
    dot_na = np.sum(n_face * a, axis=1)
    flip = dot_na < 0
    n_face[flip] *= -1
    
    # Choose circumcenter direction
    dot_un = np.sum(u * n_face, axis=1)
    cc = u.copy()
    cc[dot_un <= 0] *= -1
    
    # Compute circumradii
    dots_a = np.sum(cc * a, axis=1)
    dots_b = np.sum(cc * b, axis=1)
    dots_c = np.sum(cc * c, axis=1)
    dots_a = np.clip(dots_a, -1, 1)
    dots_b = np.clip(dots_b, -1, 1)
    dots_c = np.clip(dots_c, -1, 1)
    
    d_a = np.arccos(dots_a)
    d_b = np.arccos(dots_b)
    d_c = np.arccos(dots_c)
    crs = np.maximum(np.maximum(d_a, d_b), d_c)
    
    # Filter out invalid facets
    valid_all = valid & valid2
    if not np.all(valid_all):
        crs = crs[valid_all]
        cc = cc[valid_all]
        simplices = simplices[valid_all]
    
    if len(crs) == 0:
        return 1.0, np.zeros((n, 3))
    
    max_cr = np.max(crs)
    
    if max_cr < 1e-10:
        return max_cr, np.zeros((n, 3))
    
    # Soft-max: log(sum(exp(beta * crs))) / beta
    scaled = beta * (crs - max_cr)
    log_sum_exp = max_cr * beta + np.log(np.sum(np.exp(scaled)))
    soft_max_cr = log_sum_exp / beta
    
    # Weights for soft-max
    weights = np.exp(scaled) / np.sum(np.exp(scaled))
    
    # Compute gradient vectorized
    grad = np.zeros((n, 3))
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-15:
            continue
        cc_k = cc[k]
        for idx in simplices[k]:
            dot_val = np.dot(cc_k, pts[idx])
            dot_val = np.clip(dot_val, -1 + 1e-15, 1 - 1e-15)
            sin_d = np.sqrt(1 - dot_val * dot_val)
            if sin_d < 1e-15:
                continue
            vec = cc_k / sin_d
            vec_tan = vec - np.dot(vec, pts[idx]) * pts[idx]
            grad[idx] += w * vec_tan
    
    return soft_max_cr, grad

def _polish(pts, deadline, tol=1e-10):
    """
    High-precision polish using L-BFGS-B on a smooth soft-max covering radius objective.
    Uses tangent-space parameterization.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Try L-BFGS-B optimization
    try:
        from scipy.optimize import minimize
        
        def objective_and_grad(x):
            """Objective: soft-max covering radius. x is flattened (n,3) array."""
            p = x.reshape(n, 3)
            p = _normalize(p)
            val, grad = _soft_max_covering_radius(p, beta=50.0)
            # Project gradient to tangent space
            grad_proj = grad - np.einsum('ij,ij->i', grad, p)[:, None] * p
            return val, grad_proj.ravel()
        
        x0 = pts.ravel()
        
        # Set up bounds to keep points on unit sphere (loose bounds)
        bounds = [(-1.5, 1.5)] * (n * 3)
        
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 80,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 30
            }
        )
        
        new_pts = _normalize(result.x.reshape(n, 3))
        new_cr = _covering_radius(new_pts)
        if new_cr < best_cr:
            best_pts = new_pts
            best_cr = new_cr
        
    except Exception:
        pass
    
    # Final local polish with small steps if time allows
    if time.time() < deadline - 2:
        pts = best_pts
        step_size = 0.003
        beta = 80.0
        
        for it in range(80):
            if time.time() > deadline - 1:
                break
                
            val, grad = _soft_max_covering_radius(pts, beta=beta)
            
            if val < 1e-10:
                break
            
            # Project gradient to tangent space
            grad_proj = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
            
            # Adaptive step: scale by inverse gradient magnitude
            grad_norms = np.linalg.norm(grad_proj, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-15] = 1.0
            
            # Update with line search
            pts_new = pts.copy()
            for ls_factor in [1.0, 0.5, 0.25, 0.1]:
                cand = pts_new + step_size * ls_factor * (grad_proj / grad_norms)
                cand = _normalize(cand)
                cand_cr = _covering_radius(cand)
                if cand_cr < best_cr:
                    pts_new = cand
                    best_cr = cand_cr
                    break
            else:
                # No improvement, reduce step
                step_size *= 0.8
                continue
            
            pts = pts_new
            best_pts = pts_new.copy()
            
            if step_size < 1e-8:
                break
            
            step_size *= 0.998
        
    return _normalize(best_pts)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    restart_count = 0
    while time.time() < deadline - 30:
        restart_count += 1
        
        if restart_count == 1:
            pts = _fibonacci_points(N)
        elif restart_count == 2:
            # Jittered fibonacci for diversity
            pts = _fibonacci_points(N)
            noise = 0.2 * np.random.randn(N, 3)
            pts = _normalize(pts + noise)
        else:
            # Perturb best solution with adaptive noise
            noise_scale = 0.1 * (best_cr / 0.32)
            noise_scale = max(0.03, min(0.3, noise_scale))
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
            
        # 1. Lloyd's relaxation
        pts = _spherical_lloyd(pts, max_iter=60)
        
        # 2. Facet shrink to minimize covering radius
        remaining = deadline - time.time()
        if remaining > 40:
            pts = _optimize_facet_shrink(pts, max_iter=120, initial_step=0.35, decay=0.975)
        
        # 3. High-precision polish using L-BFGS-B on smooth surrogate
        if time.time() < deadline - 35:
            pts = _polish(pts, deadline=deadline - 10, tol=1e-10)
        
        if time.time() > deadline - 30:
            break
            
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    # Final polish on best solution if time allows
    if time.time() < deadline - 15:
        best_pts = _polish(best_pts, deadline=deadline - 3, tol=1e-11)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
