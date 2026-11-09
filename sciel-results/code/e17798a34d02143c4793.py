# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12

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

def _total_penalty(params, S, center, outer_angle):
    """Total penalty: sum of squares of containment + overlap. 0 means valid."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    total = 0.0
    for iv in inner_list:
        pen = _containment_violation(iv, outer)
        if pen > 0:
            total += pen * pen
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(inner_list[i], inner_list[j])
            if d > 0:
                total += d * d
    return total

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

def _refine_local(params, S, center, outer_angle, deadline):
    """L-BFGS-B local refinement with strict penalty."""
    def obj(p):
        return _total_penalty(p, S, center, outer_angle)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': 300, 'maxfun': 8000,
                                'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

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

def _binary_search_min_S_full(init_params_full, center, outer_angle, deadline,
                             S_lo=3.0, S_hi=6.0, max_iters=6):
    """Binary search for minimum S for full parameter packing."""
    best_S = S_hi
    best_params = None
    
    # First check if S_hi works
    p = init_params_full.copy()
    p = _refine_local(p, S_hi, center, outer_angle, deadline)
    if _check_validity(p, S_hi, center, outer_angle):
        best_params = p
        best_S = S_hi
    else:
        return None, None
    
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        S_mid = (S_lo + S_hi) / 2.0
        p = best_params.copy() if best_params is not None else init_params_full.copy()
        p = _refine_local(p, S_mid, center, outer_angle, deadline)
        if _check_validity(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p
            S_hi = S_mid
        else:
            S_lo = S_mid
    
    return best_S, best_params

def _get_voronoi_cells(S):
    """
    Generate 12 Voronoi-like cells partitioning the outer hexagon.
    Uses a symmetric arrangement: 6 cells near edges, 6 near vertices.
    Returns list of 12 cell centers (x, y) and cell boundaries.
    """
    # For a regular hexagon with side S, create 12 regions
    # Strategy: divide into 6 sectors (60° each), then split each sector
    # into an inner and outer region
    
    cells = []
    outer_verts = _hex_verts(0, 0, S, 0.0)
    
    # 6 sectors, each 60 degrees
    for sector in range(6):
        angle_center = np.radians(60 * sector)
        
        # Inner cell: closer to center
        r_inner = S * 0.4
        x_in = r_inner * np.cos(angle_center)
        y_in = r_inner * np.sin(angle_center)
        cells.append((x_in, y_in))
        
        # Outer cell: near the edge
        r_outer = S * 0.75
        x_out = r_outer * np.cos(angle_center)
        y_out = r_outer * np.sin(angle_center)
        cells.append((x_out, y_out))
    
    return cells

def _voronoi_init(S):
    """Create initial parameters using Voronoi cell centers."""
    cells = _get_voronoi_cells(S)
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = cells[i][0]
        params[3*i+1] = cells[i][1]
        params[3*i+2] = 0.0  # Start with 0 rotation
    return params

def _voronoi_refine(params, S, center, outer_angle, deadline):
    """
    Refine using Voronoi cell constraints.
    Each hexagon is constrained to stay within its assigned cell region.
    This reduces the search space and helps avoid overlaps.
    """
    cells = _get_voronoi_cells(S)
    
    def obj(p):
        # Base penalty
        pen = _total_penalty(p, S, center, outer_angle)
        
        # Add soft penalty for leaving assigned cell region
        for i in range(N):
            cx, cy = cells[i]
            # Cell radius approximately S/3
            cell_radius = S / 3.0
            dx = p[3*i] - cx
            dy = p[3*i+1] - cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > cell_radius:
                pen += (dist - cell_radius) ** 2 * 10.0
        
        return pen
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': 500, 'maxfun': 15000,
                                'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin before 120s limit

    best_S = float('inf')
    best_result = None

    center = (0.0, 0.0)
    outer_angle = 0.0

    # Generate initial configurations (2D points)
    inits_2d = []
    for nr, nc in [(2, 6), (3, 4), (4, 3), (6, 2)]:
        if nr * nc == N:
            inits_2d.append(_make_grid(nr, nc))
    inits_2d.append(_ring_init(2.0))
    inits_2d.append(_ring_init(2.5))
    inits_2d.append(_linear_init(4.0))
    inits_2d.append(_linear_init(3.5))
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        r = rng.uniform(0, 3.0, size=(N, 2))
        inits_2d.append(r - r.mean(axis=0))
        
    # Add Voronoi-based initializations
    for S_init in [4.0, 4.5, 5.0]:
        inits_2d.append(np.array(_get_voronoi_cells(S_init)))
    
    # Convert to full params (x, y, angle) with various angle inits
    inits_full = []
    for pts in inits_2d:
        for ang in [0.0, 30.0, 15.0]:
            params = np.zeros(3 * N)
            for i in range(N):
                params[3*i] = pts[i, 0]
                params[3*i+1] = pts[i, 1]
                params[3*i+2] = ang
            inits_full.append(params)

    # Try Voronoi-constrained refinement first with promising S values
    for S_try in [4.2, 4.0, 3.8, 4.5]:
        if time.time() > DEADLINE - 15:
            break
        
        # Use Voronoi init for this S
        params_voronoi = _voronoi_init(S_try)
        params_voronoi = _voronoi_refine(params_voronoi, S_try, center, outer_angle, DEADLINE)
        
        if _check_validity(params_voronoi, S_try, center, outer_angle):
            if S_try < best_S:
                best_S = S_try
                best_result = (S_try, params_voronoi, center, outer_angle)
        
        # Also try standard refinement from Voronoi init
        if time.time() < DEADLINE - 10:
            params_std = _refine_local(params_voronoi, S_try, center, outer_angle, DEADLINE)
            if _check_validity(params_std, S_try, center, outer_angle):
                if S_try < best_S:
                    best_S = S_try
                    best_result = (S_try, params_std, center, outer_angle)

    # Now try binary search from various initializations
    for init_params_full in inits_full:
        if time.time() > DEADLINE - 10:
            break
        
        S_found, params_found = _binary_search_min_S_full(
            init_params_full, center, outer_angle, DEADLINE,
            S_lo=3.0, S_hi=5.5, max_iters=6
        )
        
        if params_found is not None and S_found < best_S:
            # Final polish with tighter tolerance
            polished = _refine_local(params_found, S_found, center, outer_angle, DEADLINE)
            if _check_validity(polished, S_found, center, outer_angle):
                best_S = S_found
                best_result = (S_found, polished, center, outer_angle)
            else:
                best_S = S_found
                best_result = (S_found, params_found, center, outer_angle)

    # Final high-precision polish on best result
    if best_result is not None and time.time() < DEADLINE - 5:
        S_best, params_best, ctr, oa = best_result
        polished = _refine_local(params_best, S_best, ctr, oa, DEADLINE)
        if _check_validity(polished, S_best, ctr, oa):
            best_result = (S_best, polished, ctr, oa)

    if best_result is None:
        # Fallback: return a valid but not optimal configuration
        grid = _make_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    S, params, ctr, oa = best_result
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)

# EVOLVE-BLOCK-END
