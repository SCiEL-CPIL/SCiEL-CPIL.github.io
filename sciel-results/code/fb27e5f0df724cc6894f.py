import numpy as np
import time
import math
from scipy.spatial import ConvexHull

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy(pts):
    """
    Compute logarithmic energy.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy
    """
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    # dist_sq[i,j] = ||x_i - x_j||^2
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf')
        
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    # Note: log(1/d) = -log(d). d = sqrt(d_sq). log(d) = 0.5 log(d_sq).
    # So term is -0.5 log(d_sq).
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    return float(energy)

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy and Riemannian gradient on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: (energy, grad) where grad is the Riemannian gradient (tangent space).
    """
    n = len(pts)
    if n < 2:
        return 0.0, np.zeros_like(pts)
    
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf'), np.zeros_like(pts)

    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    
    # Euclidean gradient:
    # E = -0.25 * sum_{i,j} log(d_ij^2)
    # dE/dx_i = -0.25 * sum_{j!=i} (1/d_ij^2) * d(d_ij^2)/dx_i
    # d(d_ij^2)/dx_i = 2(x_i - x_j)
    # dE/dx_i = -0.5 * sum_{j!=i} (x_i - x_j) / d_ij^2
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    weights = -0.5 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space: grad_riem = grad_eucl - (grad_eucl . x_i) x_i
    proj_factor = np.sum(grad_eucl * pts, axis=1, keepdims=True)
    grad_riem = grad_eucl - proj_factor * pts
    
    return energy, grad_riem

def _exp_map(pts, vecs):
    """
    Exponential map on the unit sphere.
    pts: (N, 3) current positions
    vecs: (N, 3) tangent vectors
    Returns: new positions on the sphere.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-15)
    
    # If norm is very small, just add
    # Otherwise use sin(theta)/theta * vec + cos(theta) * pt
    theta = norms
    # Handle small theta limit
    sin_t_over_t = np.ones_like(theta)
    cos_t = np.ones_like(theta)
    
    mask = theta > 1e-8
    if np.any(mask):
        sin_t_over_t[mask] = np.sin(theta[mask]) / theta[mask]
        cos_t[mask] = np.cos(theta[mask])
    
    new_pts = pts * cos_t + vecs * sin_t_over_t
    
    # Re-normalize to ensure they are exactly on the sphere (numerical stability)
    new_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_norms = np.maximum(new_norms, 1e-15)
    new_pts = new_pts / new_norms
    
    return new_pts

def _retract(pts, vecs):
    """
    Simple retraction: normalize pts + vecs.
    """
    new_pts = pts + vecs
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return new_pts / norms

def _riemannian_cg(pts, max_iter=50, tol=1e-8):
    """
    Riemannian Conjugate Gradient method.
    """
    n = len(pts)
    x_k = pts.copy()
    f_k, g_k = _log_energy_and_grad(x_k)
    
    if not np.isfinite(f_k):
        return x_k, f_k
        
    # Initial direction
    p_k = -g_k
    
    for k in range(max_iter):
        # Line search: minimize f(exp_map(x_k, alpha * p_k))
        # Use simple backtracking or fixed step? 
        # Given the smoothness, a fixed small step or adaptive step is good.
        # Let's try a simple adaptive step based on gradient norm.
        
        # Normalize direction for stable step size
        p_norm = np.linalg.norm(p_k, axis=1, keepdims=True)
        p_norm = np.maximum(p_norm, 1e-15)
        p_unit = p_k / p_norm
        
        # Initial step size
        alpha = 0.1
        
        # Backtracking line search
        f_best = f_k
        x_best = x_k
        g_best = g_k
        
        for _ in range(20):
            x_try = _exp_map(x_k, alpha * p_k)
            f_try, g_try = _log_energy_and_grad(x_try)
            
            if not np.isfinite(f_try):
                alpha *= 0.5
                continue
                
            if f_try < f_best:
                f_best = f_try
                x_best = x_try
                g_best = g_try
                # Armijo condition check roughly
                # f_try <= f_k + c * alpha * <g_k, p_k>
                # We just take the first improvement for simplicity in this constrained budget
                break
            else:
                alpha *= 0.5
                
        if f_best >= f_k:
            # No improvement, try smaller step or break
            break
            
        x_k = x_best
        f_k = f_best
        g_k = g_best
        
        # Check convergence
        grad_norm = np.linalg.norm(g_k)
        if grad_norm < tol:
            break
            
        # Update direction (Polak-Ribiere)
        # beta = <g_k, g_k - g_{k-1}> / ||g_{k-1}||^2
        # Need to parallel transport g_{k-1} to x_k? 
        # For simplicity and robustness, we can just use -g_k if beta is negative or large.
        # Or simpler: Fletcher-Reeves on sphere is tricky.
        # Let's just restart CG (set p_k = -g_k) every few steps or if beta < 0.
        # Actually, for this specific problem, a simple gradient descent with adaptive step
        # or a few steps of CG with restart is often more robust than full CG with transport.
        
        # Let's use a simple momentum-like update or just restart.
        # Given the tight time limit, restarting CG (pure gradient step) is safer.
        p_k = -g_k
        
    return x_k, f_k

def _subdivided_icosahedron(n_points):
    """
    Generate points on the sphere using a subdivided icosahedron.
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 10], [0, 10, 7], [0, 7, 3],
        [0, 3, 4], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
        [7, 3, 9], [3, 4, 8], [4, 2, 6], [2, 6, 8], [6, 8, 9],
        [2, 5, 9], [2, 9, 8], [11, 4, 2], [10, 2, 6], [7, 6, 9]
    ]
    
    def _icosa_subdivide(vertices, faces, levels):
        verts_list = [v.copy() for v in vertices]
        fs = [f.copy() for f in faces]
        
        for _ in range(levels):
            new_faces = []
            mid_cache = {}
            
            def get_midpoint(v1, v2):
                idx = (min(v1, v2), max(v1, v2))
                if idx in mid_cache:
                    return mid_cache[idx]
                mid = (verts_list[v1] + verts_list[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                verts_list.append(mid)
                mid_cache[idx] = len(verts_list) - 1
                return mid_cache[idx]
                
            for f in fs:
                v1, v2, v3 = f
                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m31 = get_midpoint(v3, v1)
                
                new_faces.append([v1, m12, m31])
                new_faces.append([v2, m23, m12])
                new_faces.append([v3, m31, m23])
                new_faces.append([m12, m23, m31])
                
            fs = new_faces
            
        return np.array(verts_list), fs

    # Level 2 gives 162 points. We want 150.
    # We will generate 162 and then remove 12 points that are "redundant" or just take the first 150?
    # Taking first 150 might break symmetry slightly but is a valid start.
    # Better: Generate level 2 (162 points) and remove the 12 points that are closest to their neighbors?
    # Or just use the 162 points, optimize, and then... wait, N must be 150.
    # So we must start with exactly 150.
    # Let's generate level 2 (162) and drop the 12 points that have the smallest local separation?
    # Or simpler: Generate level 1 (42 points) and subdivide specific faces?
    # Actually, for N=150, a "Fibonacci" or "Random" start is often better than a strict subset of a geodesic dome
    # because the geodesic dome has fixed topology that might not be optimal for log energy.
    # However, the prompt asks for a geodesic dome strategy.
    # Let's generate 162 points, and then greedily remove 12 points that are closest to another point.
    
    verts, _ = _icosa_subdivide(verts, faces, 2)
    
    if len(verts) > n_points:
        # Remove points that are closest to their nearest neighbor
        # Calculate min distance for each point
        min_dists = np.zeros(len(verts))
        for i in range(len(verts)):
            dists = np.linalg.norm(verts - verts[i], axis=1)
            np.fill_diagonal(dists, np.inf)
            min_dists[i] = np.min(dists)
        
        # Sort by min_dist, remove the smallest ones
        order = np.argsort(min_dists)
        indices_to_keep = order[len(verts)-n_points:]
        verts = verts[indices_to_keep]
        
    return verts

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Geodesic Dome (Subdivided Icosahedron)
    try:
        geo_pts = _subdivided_icosahedron(N)
        inits.append(geo_pts)
    except Exception:
        pass
        
    # 2. Fibonacci Sphere
    inits.append(_fibonacci_sphere(N))
    
    # 3. Random Uniform
    for seed in [42, 123, 999]:
        inits.append(_random_uniform_sphere(N, seed=seed))
        
    # Optimization Loop using Riemannian CG
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
            
        try:
            opt_pts, e = _riemannian_cg(init_pts, max_iter=100)
            
            # Normalize
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts
                
        except Exception:
            continue
            
    # If we have time, do a final polish on the best result with more iterations
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        try:
            opt_pts, e = _riemannian_cg(best_pts, max_iter=200, tol=1e-10)
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
