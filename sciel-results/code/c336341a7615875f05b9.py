"""Unit regular hexagons inside a regular hexagon n=12, minimize outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _polygon_overlap_depth(A, B):
    """SAT overlap depth between convex polygons A and B (arrays of 2D verts). 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = np.linalg.norm(edge)
            if length > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / length)
    for ax in axes:
        pa = np.dot(A, ax)
        pb = np.dot(B, ax)
        if pa.max() <= pb.min() or pb.max() <= pa.min():
            return 0.0
        ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if ov < min_depth:
            min_depth = ov
    return min_depth

def _containment_violation(inner_verts, outer_verts):
    """Sum of distances by which inner verts fall outside outer (edge-wise)."""
    n_out = len(outer_verts)
    pen = 0.0
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-12:
                continue
            nrm = np.array([-edge[1], edge[0]]) / el
            sd = np.dot(iv - v1, nrm)
            if sd < 0:
                pen += -sd
    return pen

def _smooth_max(x, beta=10.0):
    """Smooth approximation of max(0, x) using softplus."""
    return np.log1p(np.exp(beta * x)) / beta

def _total_penalty(params, S, center, outer_angle, smooth_beta=10.0):
    """Total penalty: sum of squares of containment + overlap. 0 means valid."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    total = 0.0
    for iv in inner_list:
        pen = _containment_violation(iv, outer)
        if pen > 0:
            # Use smooth penalty to help gradient-based optimizers
            total += _smooth_max(pen, smooth_beta) ** 2
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(inner_list[i], inner_list[j])
            if d > 0:
                total += _smooth_max(d, smooth_beta) ** 2
    return total

def _total_penalty_xy(params, S, center, outer_angle, smooth_beta=10.0):
    """Total penalty for fixed angle 0. params is [x0, y0, x1, y1, ...]"""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    total = 0.0
    for iv in inner_list:
        pen = _containment_violation(iv, outer)
        if pen > 0:
            total += _smooth_max(pen, smooth_beta) ** 2
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(inner_list[i], inner_list[j])
            if d > 0:
                total += _smooth_max(d, smooth_beta) ** 2
    return total

def _check_validity_xy(params, S, center, outer_angle, tol=1e-6):
    """Check if packing is valid (no overlaps, all contained) for fixed angle 0."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    for iv in inner_list:
        if _containment_violation(iv, outer) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _polygon_overlap_depth(inner_list[i], inner_list[j]) > tol:
                return False
    return True

def _check_validity(params, S, center, outer_angle, tol=1e-6):
    """Check if packing is valid (no overlaps, all contained)."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    for iv in inner_list:
        if _containment_violation(iv, outer) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _polygon_overlap_depth(inner_list[i], inner_list[j]) > tol:
                return False
    return True

def _make_grid(nrows, ncols):
    pos = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def _ring_init(radius=2.0):
    pts = []
    for i in range(6):
        a = np.radians(60*i)
        pts.append([radius*np.cos(a), radius*np.sin(a)])
    for i in range(6):
        a = np.radians(60*i + 30)
        pts.append([(radius*1.5)*np.cos(a), (radius*1.5)*np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def _linear_init(S):
    xs = np.linspace(-S*0.7, S*0.7, N)
    return np.column_stack([xs, np.zeros(N)])

def _refine_local_xy(params, S, center, outer_angle, deadline, high_precision=False):
    """L-BFGS-B local refinement for fixed angle 0."""
    def obj(p):
        return _total_penalty_xy(p, S, center, outer_angle)
    
    if high_precision:
        options = {'maxiter': 1000, 'maxfun': 50000,
                   'ftol': 1e-16, 'gtol': 1e-12}
    else:
        options = {'maxiter': 500, 'maxfun': 10000,
                   'ftol': 1e-14, 'gtol': 1e-10}
    
    try:
        res = minimize(obj, params, method='L-BFGS-B', options=options)
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _refine_local(params, S, center, outer_angle, deadline, high_precision=False):
    """L-BFGS-B local refinement with strict penalty."""
    def obj(p):
        return _total_penalty(p, S, center, outer_angle)
    
    if high_precision:
        options = {'maxiter': 1000, 'maxfun': 50000,
                   'ftol': 1e-16, 'gtol': 1e-12}
    else:
        options = {'maxiter': 500, 'maxfun': 10000,
                   'ftol': 1e-14, 'gtol': 1e-10}
    
    try:
        res = minimize(obj, params, method='L-BFGS-B', options=options)
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _binary_search_min_S_xy(init_params_xy, center, outer_angle, deadline,
                             S_lo=3.0, S_hi=6.0, max_iters=10):
    """Binary search for minimum S for fixed angle 0 packing."""
    best_S = S_hi
    best_params = None
    
    # First check if S_hi works
    p = init_params_xy.copy()
    p = _refine_local_xy(p, S_hi, center, outer_angle, deadline)
    if _check_validity_xy(p, S_hi, center, outer_angle):
        best_params = p
        best_S = S_hi
    else:
        return None, None
    
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        S_mid = (S_lo + S_hi) / 2.0
        p = best_params.copy() if best_params is not None else init_params_xy.copy()
        p = _refine_local_xy(p, S_mid, center, outer_angle, deadline)
        if _check_validity_xy(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p
            S_hi = S_mid
        else:
            S_lo = S_mid
    
    # Final high-precision polish
    if time.time() < deadline - 2 and best_params is not None:
        p_polished = _refine_local_xy(best_params, best_S, center, outer_angle, deadline, high_precision=True)
        if _check_validity_xy(p_polished, best_S, center, outer_angle, tol=1e-8):
            best_params = p_polished
    
    return best_S, best_params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin before 120s limit

    best_S = float('inf')
    best_result = None

    centers = [(0.0, 0.0)]
    # More angles to explore - focus on promising ones first
    outer_angles = [0.0, 30.0, 60.0, 15.0, 45.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0]

    # Generate initial configurations
    inits = []
    for nr, nc in [(2, 6), (3, 4), (4, 3), (6, 2)]:
        if nr * nc == N:
            inits.append(_make_grid(nr, nc))
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    inits.append(_linear_init(4.0))
    inits.append(_linear_init(3.5))
    
    # Add more diverse initial configurations
    rng = np.random.default_rng(42)
    for _ in range(12):
        r = rng.uniform(0, 3.0, size=(N, 2))
        inits.append(r - r.mean(axis=0))
    
    # Add hexagonal lattice-based initial configs
    for offset in [0.0, 0.5, 1.0]:
        pts = []
        for i in range(6):
            a = np.radians(60*i + 30)
            pts.append([2.0*np.cos(a) + offset*0.3, 2.0*np.sin(a) + offset*0.3])
        for i in range(6):
            a = np.radians(60*i)
            pts.append([3.5*np.cos(a) - offset*0.2, 3.5*np.sin(a) - offset*0.2])
        inits.append(np.array(pts) - np.array(pts).mean(axis=0))

    for oa in outer_angles:
        if time.time() > DEADLINE - 20:
            break
        for ctr in centers:
            if time.time() > DEADLINE - 15:
                break
            for init_pts in inits:
                if time.time() > DEADLINE - 10:
                    break
                
                # 1. Try fixed-angle optimization (faster, often optimal for hexagons)
                init_params_xy = np.zeros(2 * N)
                for i in range(N):
                    init_params_xy[2*i] = init_pts[i, 0]
                    init_params_xy[2*i+1] = init_pts[i, 1]
                
                S_found_xy, params_found_xy = _binary_search_min_S_xy(
                    init_params_xy, ctr, oa, DEADLINE,
                    S_lo=3.0, S_hi=5.5, max_iters=8
                )
                
                if params_found_xy is not None and S_found_xy < best_S:
                    # Convert to full params for result
                    params_full = np.zeros(3 * N)
                    for i in range(N):
                        params_full[3*i] = params_found_xy[2*i]
                        params_full[3*i+1] = params_found_xy[2*i+1]
                        params_full[3*i+2] = 0.0
                    
                    # Final polish with full params (allowing slight angle adjustment if needed)
                    polished = _refine_local(params_full, S_found_xy, ctr, oa, DEADLINE)
                    if _check_validity(polished, S_found_xy, ctr, oa):
                        best_S = S_found_xy
                        best_result = (S_found_xy, polished, ctr, oa)
                    else:
                        best_S = S_found_xy
                        best_result = (S_found_xy, params_full, ctr, oa)

                # Early exit if we found a very good solution
                if best_S < 3.3:
                    break
            if time.time() > DEADLINE - 10:
                break
        if time.time() > DEADLINE - 10:
            break

    if best_result is None:
        # Fallback: return a valid but not optimal configuration
        grid = _make_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    S, params, ctr, oa = best_result
    
    # Final high-precision polish pass
    if time.time() < DEADLINE - 5:
        # First, try high-precision refinement at current S
        p_hp = _refine_local(params, S, ctr, oa, DEADLINE, high_precision=True)
        if _check_validity(p_hp, S, ctr, oa, tol=1e-8):
            params = p_hp
        
        # Try to reduce S in small steps
        for factor in [0.995, 0.99, 0.985, 0.98]:
            if time.time() > DEADLINE - 3:
                break
            S_try = S * factor
            p_try = params.copy()
            p_try = _refine_local(p_try, S_try, ctr, oa, DEADLINE, high_precision=True)
            if _check_validity(p_try, S_try, ctr, oa, tol=1e-8):
                S = S_try
                params = p_try
                best_S = S_try
                best_result = (S_try, p_try, ctr, oa)

    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)

# EVOLVE-BLOCK-END
