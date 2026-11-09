"""Unit regular hexagons inside a regular hexagon n=12, minimize outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Generate vertices of a regular hexagon (pointy-top by default)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _polygon_overlap_depth(A, B):
    """SAT overlap depth between convex polygons A and B. Returns 0 if disjoint."""
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
    """Sum of distances by which inner verts fall outside outer."""
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

def _check_validity(params, S, center, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all contained."""
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

def _min_enclosing_S(all_points, outer_angle_deg):
    """
    Compute minimum S such that all_points fit inside a regular hexagon 
    of side S, centered at origin, with given orientation.
    """
    # Center the points
    centroid = all_points.mean(axis=0)
    centered = all_points - centroid
    
    outer_angle = np.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = outer_angle + np.radians(30) + np.radians(60) * k
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    
    max_range = 0.0
    for nrm in normals:
        projections = np.dot(centered, nrm)
        rng = projections.max() - projections.min()
        if rng > max_range:
            max_range = rng
    
    return max_range / np.sqrt(3.0)

def _objective(params, outer_angle_deg):
    """
    Objective: minimize enclosing hexagon side S + overlap penalty.
    params: [x0, y0, a0, x1, y1, a1, ...] for N hexagons
    """
    all_points = []
    for i in range(N):
        verts = _hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2])
        all_points.append(verts)
    
    all_points = np.vstack(all_points)
    
    S = _min_enclosing_S(all_points, outer_angle_deg)
    
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(all_points[i*6:(i+1)*6], all_points[j*6:(j+1)*6])
            if d > 0:
                overlap_pen += d * d
    
    return S + 100.0 * overlap_pen

def _hex_lattice_init(spacing=np.sqrt(3.0)):
    """Generate hexagonal lattice points and select a compact cluster of N."""
    points = []
    for j in range(-4, 5):
        for i in range(-4, 5):
            x = i * spacing + (j % 2) * spacing / 2
            y = j * spacing * np.sqrt(3) / 2
            points.append([x, y])
    points = np.array(points)
    
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)[:N]
    selected = points[idx]
    selected -= selected.mean(axis=0)
    return selected

def _cluster_init():
    """Create a hexagonal cluster: 1 center + 6 first ring + 5 second ring."""
    pts = []
    pts.append([0.0, 0.0])
    d1 = np.sqrt(3.0)
    for i in range(6):
        a = np.radians(60 * i)
        pts.append([d1 * np.cos(a), d1 * np.sin(a)])
    d2 = 2 * d1
    for i in range(6):
        a = np.radians(60 * i + 30)
        pts.append([d2 * np.cos(a), d2 * np.sin(a)])
    pts = np.array(pts[:12])
    pts -= pts.mean(axis=0)
    return pts

def _linear_init():
    """Create a 2-row linear arrangement."""
    pts = []
    for i in range(6):
        pts.append([(i - 2.5) * np.sqrt(3), 0.0])
    for i in range(6):
        pts.append([(i - 2.5) * np.sqrt(3) + np.sqrt(3)/2, np.sqrt(3) * 1.5])
    pts = np.array(pts[:12])
    pts -= pts.mean(axis=0)
    return pts

def _optimize_for_S(init_pts, outer_angle_deg, deadline):
    """Optimize positions and angles to minimize enclosing hexagon side."""
    params0 = np.zeros(3 * N)
    for i in range(N):
        params0[3*i] = init_pts[i, 0]
        params0[3*i+1] = init_pts[i, 1]
        params0[3*i+2] = 0.0
    
    best_params = params0.copy()
    best_S = float('inf')
    
    # Compute initial S
    all_pts0 = np.vstack([_hex_verts(params0[3*i], params0[3*i+1], 1.0, params0[3*i+2]) for i in range(N)])
    S0 = _min_enclosing_S(all_pts0, outer_angle_deg)
    if _check_validity(params0, S0, (0.0, 0.0), outer_angle_deg):
        best_S = S0
    
    # L-BFGS-B optimization
    try:
        res = minimize(
            _objective, params0,
            args=(outer_angle_deg,),
            method='L-BFGS-B',
            options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        p = res.x
        all_pts = np.vstack([_hex_verts(p[3*i], p[3*i+1], 1.0, p[3*i+2]) for i in range(N)])
        S = _min_enclosing_S(all_pts, outer_angle_deg)
        if _check_validity(p, S, (0.0, 0.0), outer_angle_deg) and S < best_S:
            best_params = p
            best_S = S
    except Exception:
        pass
    
    return best_S, best_params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0

    best_S = float('inf')
    best_result = None

    outer_angles = [0.0, 30.0]
    
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_cluster_init())
    inits.append(_linear_init())
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        r = rng.uniform(-2.0, 2.0, size=(N, 2))
        inits.append(r - r.mean(axis=0))

    for oa in outer_angles:
        if time.time() > DEADLINE - 30:
            break
        for init_pts in inits:
            if time.time() > DEADLINE - 15:
                break
            
            S_found, params_found = _optimize_for_S(init_pts, oa, DEADLINE)
            
            if params_found is not None and S_found < best_S:
                best_S = S_found
                best_result = (S_found, params_found, oa)
            
            if best_S < 3.0:
                break
        if time.time() > DEADLINE - 15:
            break

    if best_result is None:
        grid = _hex_lattice_init()
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    S, params, oa = best_result
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (0.0, 0.0), float(S), float(oa)

# EVOLVE-BLOCK-END
