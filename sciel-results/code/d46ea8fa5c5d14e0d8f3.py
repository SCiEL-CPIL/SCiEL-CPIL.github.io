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
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
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
        n = n / np.linalg.norm(n)
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

def _jittered_fibonacci(n, seed=42, jitter=0.1):
    """Fibonacci lattice with random jitter to break symmetry."""
    np.random.seed(seed)
    pts = _fibonacci_points(n)
    noise = jitter * np.random.randn(n, 3)
    pts = pts + noise
    return _normalize(pts)

def _spherical_lloyd(pts, max_iter=200, tol=1e-9):
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

def _facet_circumcenter(pts, simplex):
    """Compute circumcenter and circumradius of a spherical triangle facet."""
    a = pts[simplex[0]]
    b = pts[simplex[1]]
    c = pts[simplex[2]]
    
    v = np.cross(a - b, b - c)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        if d_ab >= d_bc and d_ab >= d_ca:
            cc = (a + b) / np.linalg.norm(a + b)
        elif d_bc >= d_ab and d_bc >= d_ca:
            cc = (b + c) / np.linalg.norm(b + c)
        else:
            cc = (c + a) / np.linalg.norm(c + a)
        cr = max(d_ab, d_bc, d_ca) / 2.0
        return cc, cr
    
    u = v / norm_v
    
    n_face = np.cross(b - a, c - a)
    n_face = n_face / np.linalg.norm(n_face)
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
    
    return cc, cr

def _soft_covering_radius(pts, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp (soft-max).
    Returns (soft_cr, active_weights, facet_crs, circumcenters)
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius(pts), None, None, None
    
    n_facets = len(hull.simplices)
    crs = np.zeros(n_facets)
    ccs = [None] * n_facets
    
    for fi, simplex in enumerate(hull.simplices):
        cc, cr = _facet_circumcenter(pts, simplex)
        ccs[fi] = cc
        crs[fi] = cr
    
    # Soft-max: (1/beta) * log(sum(exp(beta * cr_i)))
    # For numerical stability, subtract max
    max_cr = np.max(crs)
    if max_cr < 1e-10:
        return max_cr, np.ones(n_facets) / n_facets, crs, ccs
    
    scaled = beta * (crs - max_cr)
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    exps = np.exp(scaled)
    log_sum = np.log(np.sum(exps))
    soft_cr = max_cr + log_sum / beta
    
    # Weights: softmax(beta * crs)
    weights = exps / np.sum(exps)
    
    return soft_cr, weights, crs, ccs

def _soft_covering_gradient(pts, beta=50.0):
    """
    Compute the gradient of the soft covering radius w.r.t. points.
    Returns (soft_cr, grad) where grad has shape (n, 3).
    """
    pts = _normalize(pts)
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius(pts), np.zeros_like(pts)
    
    n_facets = len(hull.simplices)
    crs = np.zeros(n_facets)
    ccs = [None] * n_facets
    
    for fi, simplex in enumerate(hull.simplices):
        cc, cr = _facet_circumcenter(pts, simplex)
        ccs[fi] = cc
        crs[fi] = cr
    
    max_cr = np.max(crs)
    if max_cr < 1e-10:
        return max_cr, np.zeros_like(pts)
    
    # Soft-max weights
    scaled = beta * (crs - max_cr)
    scaled = np.clip(scaled, -50, 50)
    exps = np.exp(scaled)
    weights = exps / np.sum(exps)
    
    # Gradient: sum over facets of weight_i * d(cr_i)/d(pts)
    # For each facet, cr_i = max over vertices of arccos(dot(cc, p_v))
    # d(arccos(t))/dt = -1/sqrt(1-t^2)
    # d(arccos(dot(cc, p)))/d(p) = -cc / sqrt(1 - dot(cc,p)^2)
    
    grad = np.zeros_like(pts)
    
    for fi in range(n_facets):
        if weights[fi] < 1e-10:
            continue
        
        simplex = hull.simplices[fi]
        cc = ccs[fi]
        
        for idx in simplex:
            p = pts[idx]
            dot_val = np.clip(np.dot(cc, p), -1, 1)
            denom = np.sqrt(max(1e-15, 1.0 - dot_val * dot_val))
            # Gradient contribution: -cc / denom, weighted
            grad[idx] += weights[fi] * (-cc / denom)
    
    # Project onto tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return max_cr + np.log(np.sum(exps)) / beta, grad

def _gradient_descent_soft(pts, max_iter=500, beta=50.0, lr=0.05, tol=1e-10):
    """
    Gradient descent using the smooth soft-max surrogate.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        soft_cr, grad = _soft_covering_gradient(pts, beta)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Adaptive step size
        step = lr / max(1.0, grad_norm)
        
        # Line search
        best_step = step
        best_cr = soft_cr
        
        for factor in [1.0, 0.5, 0.25, 0.1, 0.05]:
            trial = pts - factor * step * grad
            trial = _normalize(trial)
            trial_cr, _ = _soft_covering_gradient(trial, beta)
            if trial_cr < best_cr:
                best_cr = trial_cr
                best_step = factor * step
        
        pts = _normalize(pts - best_step * grad)
        
        # Reduce learning rate slowly
        lr *= 0.995
        
        if it % 50 == 0:
            exact_cr = _covering_radius(pts)
            if exact_cr < tol:
                break
        
    return pts

def _high_precision_polish(pts, max_iter=200, tol=1e-12):
    """
    High-precision polish: iteratively adjust points to minimize the worst facet circumradius.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_cr = 0.0
        worst_facet = None
        worst_cc = None
        
        for simplex in hull.simplices:
            cc, cr = _facet_circumcenter(pts, simplex)
            if cr > max_cr:
                max_cr = cr
                worst_facet = simplex
                worst_cc = cc
        
        if worst_facet is None or max_cr < tol:
            break
        
        step_size = 0.1 * max_cr
        
        for idx in worst_facet:
            p = pts[idx]
            vec = worst_cc - p
            vec_tan = vec - np.dot(vec, p) * p
            norm_vec = np.linalg.norm(vec_tan)
            if norm_vec > 1e-12:
                pts[idx] = p + step_size * (vec_tan / norm_vec)
        
        pts = _normalize(pts)
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    best_pts = _jittered_fibonacci(N, seed=42, jitter=0.05)
    best_cr = _covering_radius(best_pts)
    
    n_restarts = 7
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        if restart == 0:
            pts = _jittered_fibonacci(N, seed=42, jitter=0.05)
        elif restart == 1:
            pts = _fibonacci_points(N)
        elif restart == 2:
            vecs = np.random.randn(N, 3)
            pts = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        elif restart == 3:
            # Icosahedron-based
            phi = (1 + np.sqrt(5)) / 2
            base = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
            ], dtype=float)
            base = _normalize(base)
            # Expand to 50 points by adding jittered copies
            pts = np.tile(base, (5, 1))
            noise = 0.3 * np.random.randn(N, 3)
            pts = _normalize(pts + noise)
        elif restart == 4:
            # Different jitter level
            pts = _jittered_fibonacci(N, seed=123, jitter=0.15)
        elif restart == 5:
            # Another random init
            vecs = np.random.randn(N, 3)
            pts = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        else:
            noise_scale = 0.15 / (1 + (restart - 6) * 0.2)
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
        
        # Stage 1: Lloyd's for global structure
        pts = _spherical_lloyd(pts, max_iter=100)
        
        # Stage 2: Smooth gradient descent with increasing beta
        if time.time() < deadline - 20:
            # Beta cascade: start smooth, then sharpen
            for beta in [15, 50, 150, 300]:
                if time.time() > deadline - 15:
                    break
                pts = _gradient_descent_soft(pts, max_iter=150, beta=beta, lr=0.05, tol=1e-10)
        
        # Stage 3: High-precision polish
        if time.time() < deadline - 10:
            pts = _high_precision_polish(pts, max_iter=150, tol=1e-12)
        
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
