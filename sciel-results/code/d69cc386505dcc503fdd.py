"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def _bary_to_cart(bary):
    """Convert barycentric coords (a,b,c) with a+b+c=1 to cartesian."""
    return bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2

def _cart_to_bary(pts):
    """Convert cartesian points to barycentric coords."""
    n = len(pts)
    bary = np.zeros((n, 3))
    for i in range(n):
        px, py = pts[i]
        # Solve: p = a*V0 + b*V1 + c*V2, a+b+c=1
        # px = b + 0.5*c
        # py = sqrt(3)/2 * c
        c = 2 * py / np.sqrt(3)
        b = px - 0.5 * c
        a = 1 - b - c
        bary[i] = [a, b, c]
    return bary

def _clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to be non-negative and sum to 1."""
    bary = np.maximum(bary, 0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    bary = bary / s
    return bary

def _soft_min_area(pts, beta=50.0):
    """
    Smooth approximation of the minimum triangle area using soft-min.
    Returns (value, gradient) where gradient is w.r.t. the flat point coordinates.
    """
    n = len(pts)
    areas = []
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) -
                                  (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]))
                areas.append(area)
                triples.append((i, j, k))
    
    areas = np.array(areas)
    m = len(areas)
    
    # Soft min: -log(sum(exp(-beta*a))) / beta
    # This is smooth and approximates min(areas) as beta -> inf
    max_area = np.max(areas)
    exp_vals = np.exp(-beta * (areas - max_area))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = max_area - log_sum / beta
    
    # Gradient: d(soft_min)/d(area_m) = exp(-beta*(area_m - max)) / sum(exp(-beta*(area_l - max)))
    # = exp_vals[m] / sum(exp_vals)
    weights = exp_vals / np.sum(exp_vals)
    
    # Now compute gradient w.r.t. point coordinates
    # d(area_m)/d(pts) for each triangle
    grad = np.zeros((n, 2))
    for idx, (i, j, k) in enumerate(triples):
        w = weights[idx]
        if w < 1e-15:
            continue
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        cross_val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1])
        sign_cross = np.sign(cross_val)
        if sign_cross == 0:
            sign_cross = 1.0
        
        # Gradient of area = 0.5 * sign * grad(cross)
        g_pi = 0.5 * sign_cross * np.array([pi[1] - pk[1], pi[0] - pj[0]])
        g_pj = 0.5 * sign_cross * np.array([pi[0] - pk[0], pk[1] - pi[1]])
        g_pk = 0.5 * sign_cross * np.array([pi[1] - pj[1], pj[0] - pi[0]])
        
        grad[i] += w * g_pi
        grad[j] += w * g_pj
        grad[k] += w * g_pk
    
    return soft_min_val, grad

def _objective_bary(bary_flat, beta=50.0):
    """
    Objective in barycentric coordinates.
    bary_flat is shape (3*n,), each group of 3 is (a,b,c) for a point.
    We maximize the soft-min triangle area, so we minimize negative of it.
    """
    bary = bary_flat.reshape(-1, 3)
    bary_clamped = _clamp_bary(bary)
    pts = _bary_to_cart(bary_clamped)
    soft_min_val, grad_pts = _soft_min_area(pts, beta)
    
    # Chain rule: d(soft_min)/d(bary) via d(pts)/d(bary)
    # pts = a*V0 + b*V1 + c*V2
    # d(pts[i])/d(bary[i,0]) = V0, d(pts[i])/d(bary[i,1]) = V1, d(pts[i])/d(bary[i,2]) = V2
    grad_bary = np.zeros_like(bary_clamped)
    for i in range(len(pts)):
        grad_bary[i, 0] = np.dot(grad_pts[i], V0)
        grad_bary[i, 1] = np.dot(grad_pts[i], V1)
        grad_bary[i, 2] = np.dot(grad_pts[i], V2)
    
    # We want to maximize, so return negative
    return -soft_min_val, -grad_bary.flatten()

def _objective_bary_value_only(bary_flat, beta=50.0):
    """Value-only objective for Nelder-Mead."""
    bary = bary_flat.reshape(-1, 3)
    bary_clamped = _clamp_bary(bary)
    pts = _bary_to_cart(bary_clamped)
    soft_min_val, _ = _soft_min_area(pts, beta)
    return -soft_min_val

def _make_initial_bary_grid(jitter=0.0, seed=0):
    """
    Generate structured initial barycentric coordinates using a grid-like pattern.
    Uses a quasi-random or structured placement.
    """
    n = N
    rng = np.random.RandomState(seed)
    
    # Create a structured grid in the triangle
    # Use barycentric grid: for a triangle, we can use a triangular lattice
    # For 11 points, use a pattern based on 3-4-4 or similar
    
    # Method: place points at vertices, edge midpoints, and interior
    bary = np.zeros((n, 3))
    
    # Vertex regions (slightly inside)
    eps_in = 0.08
    bary[0] = [1 - eps_in, eps_in/2, eps_in/2]  # near V0
    bary[1] = [eps_in/2, 1 - eps_in, eps_in/2]  # near V1
    bary[2] = [eps_in/2, eps_in/2, 1 - eps_in]  # near V2
    
    # Edge midpoints (slightly inside)
    eps_edge = 0.05
    bary[3] = [1 - 2*eps_edge, eps_edge, eps_edge]  # midpoint of V0-V1, pushed in
    bary[4] = [eps_edge, 1 - 2*eps_edge, eps_edge]  # midpoint of V1-V2
    bary[5] = [eps_edge, eps_edge, 1 - 2*eps_edge]  # midpoint of V0-V2
    
    # Interior points - use a small grid
    # For 5 more points, place them in a pattern
    # Use barycentric coordinates that form a smaller triangle pattern
    interior_configs = [
        [1/3, 1/3, 1/3],  # centroid
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4],
        [0.5, 0.25, 0.25],
    ]
    for i in range(5):
        bary[6 + i] = interior_configs[i]
    
    # Add jitter if specified
    if jitter > 0:
        noise = rng.uniform(-jitter, jitter, size=bary.shape)
        bary += noise
        bary = _clamp_bary(bary)
    
    return bary

def _make_initial_bary_random(seed=0):
    """Generate random initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    # Random points in the triangle
    u = rng.random((N, 1))
    v = rng.random((N, 1))
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    
    bary = np.zeros((N, 3))
    bary[:, 0] = 1 - u[:, 0] - v[:, 0]
    bary[:, 1] = u[:, 0]
    bary[:, 2] = v[:, 0]
    
    # Ensure points are not too close to boundary
    bary = _clamp_bary(bary)
    
    return bary

def _make_initial_bary_perturb(base_bary, jitter, seed=0):
    """Perturb a base configuration."""
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=base_bary.shape)
    bary = base_bary + noise
    bary = _clamp_bary(bary)
    return bary

def _local_optimize_bary(bary_flat, beta=50.0, maxiter=500):
    """
    Local optimization in barycentric space using L-BFGS-B.
    Returns the optimized barycentric coordinates.
    """
    # Bounds: each barycentric coord in [0, 1]
    bounds = [(0, 1)] * (3 * N)
    
    result = minimize(
        _objective_bary,
        bary_flat,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        args=(beta,),
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
    )
    
    return result.x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    best_bary = None
    
    # Phase 1: Grid-based initialization with structured jitter
    # Generate diverse starting points from the grid pattern
    base_bary = _make_initial_bary_grid(jitter=0.0, seed=0)
    
    # Create multiple perturbed versions
    initial_configs = []
    
    # Original grid
    initial_configs.append(base_bary.copy())
    
    # Jittered versions
    for j in range(1, 8):
        jitter_amt = 0.05 * j
        config = _make_initial_bary_perturb(base_bary, jitter_amt, seed=j)
        initial_configs.append(config)
    
    # Random configurations
    for r in range(5):
        config = _make_initial_bary_random(seed=100 + r)
        initial_configs.append(config)
    
    # Phase 2: Multi-start optimization
    for idx, bary_config in enumerate(initial_configs):
        if time.time() > deadline - 30.0:
            break
        
        bary_flat = bary_config.flatten()
        
        # Try L-BFGS-B with smooth objective
        try:
            # Use a moderate beta for smoothness
            beta_val = 30.0
            optimized_bary_flat = _local_optimize_bary(bary_flat, beta=beta_val, maxiter=300)
            
            if time.time() > deadline - 15.0:
                break
            
            # Evaluate with exact min (not soft-min)
            bary_opt = _clamp_bary(optimized_bary_flat.reshape(-1, 3))
            pts_opt = _bary_to_cart(bary_opt)
            score = min_triangle_area(pts_opt)
            
            if score > best_score:
                best_score = score
                best_points = pts_opt.copy()
                best_bary = bary_opt.copy()
                
        except Exception:
            pass
        
        # Try Nelder-Mead from this starting point
        if time.time() > deadline - 10.0:
            break
        
        try:
            beta_nm = 20.0
            result_nm = minimize(
                _objective_bary_value_only,
                bary_flat,
                method='Nelder-Mead',
                args=(beta_nm,),
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            bary_nm = _clamp_bary(result_nm.x.reshape(-1, 3))
            pts_nm = _bary_to_cart(bary_nm)
            score_nm = min_triangle_area(pts_nm)
            
            if score_nm > best_score:
                best_score = score_nm
                best_points = pts_nm.copy()
                best_bary = bary_nm.copy()
                
        except Exception:
            pass
    
    # Phase 3: Final polish - use higher beta for more precise optimization
    if best_bary is not None and time.time() < deadline - 10.0:
        try:
            beta_high = 100.0
            polished_bary_flat = _local_optimize_bary(best_bary.flatten(), beta=beta_high, maxiter=500)
            bary_polished = _clamp_bary(polished_bary_flat.reshape(-1, 3))
            pts_polished = _bary_to_cart(bary_polished)
            score_polished = min_triangle_area(pts_polished)
            
            if score_polished > best_score:
                best_score = score_polished
                best_points = pts_polished.copy()
                best_bary = bary_polished.copy()
                
        except Exception:
            pass
    
    # Phase 4: One more polish with even tighter settings if time permits
    if best_bary is not None and time.time() < deadline - 5.0:
        try:
            beta_very_high = 200.0
            polished_bary_flat = _local_optimize_bary(best_bary.flatten(), beta=beta_very_high, maxiter=300)
            bary_polished = _clamp_bary(polished_bary_flat.reshape(-1, 3))
            pts_polished = _bary_to_cart(bary_polished)
            score_polished = min_triangle_area(pts_polished)
            
            if score_polished > best_score:
                best_score = score_polished
                best_points = pts_polished.copy()
                best_bary = bary_polished.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: use the grid configuration
        bary = _make_initial_bary_grid(jitter=0.0, seed=0)
        best_points = _bary_to_cart(bary)
    
    return best_points
# EVOLVE-BLOCK-END
