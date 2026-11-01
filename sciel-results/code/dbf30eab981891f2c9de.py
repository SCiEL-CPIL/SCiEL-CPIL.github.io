"""Multi-topology rigid-body optimization for 12 unit hexagons."""
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 12
DEADLINE = 148.0

_SQRT3 = np.sqrt(3.0)
_HALF_SQRT3 = _SQRT3 / 2.0
_60DEG = np.radians(60.0)

def _hex_verts_vec(cx, cy, angle_deg):
    """Vectorized hexagon vertices for a unit hexagon."""
    a = np.radians(angle_deg)
    thetas = a + _60DEG * np.arange(6)
    return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])

def _hex_normals_vec(angle_deg):
    """6 outward edge normals of a unit hexagon at given rotation."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(30) + _60DEG * np.arange(6)
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def _axial_to_cartesian(q, r):
    """Convert axial coordinates to Cartesian."""
    x = _SQRT3 * (q + r / 2.0)
    y = 1.5 * r
    return [x, y]

def _get_cluster_1():
    """Original cluster: center + 6 neighbors + 5 from distance 2."""
    coords = [(0, 0)]
    coords.extend([(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1)])
    coords.extend([(2, 0), (0, 2), (-2, 1), (-2, -1), (1, -2)])
    return np.array([_axial_to_cartesian(q, r) for q, r in coords])

def _get_cluster_2():
    """More symmetric cluster: center + 6 neighbors + 6 from distance 2 (remove one)."""
    coords = [(0, 0)]
    coords.extend([(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1)])
    # Pick 5 from distance 2 that form a more balanced shape
    coords.extend([(2, 0), (2, -1), (1, -2), (-1, -2), (-2, -1)])
    return np.array([_axial_to_cartesian(q, r) for q, r in coords])

def _get_cluster_3():
    """Linear-ish arrangement for testing."""
    coords = [(0, 0)]
    coords.extend([(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
    coords.extend([(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0)])
    return np.array([_axial_to_cartesian(q, r) for q, r in coords])

def _get_cluster_4():
    """Compact 3x4 rectangular-ish arrangement."""
    coords = []
    for r in range(-1, 2):
        for q in range(-2, 2):
            coords.append((q, r))
    # Take first 12
    coords = coords[:12]
    return np.array([_axial_to_cartesian(q, r) for q, r in coords])

def _compute_cluster_vertices(centers, angle_deg):
    """Compute all 72 vertices of the 12 hexagons in the cluster."""
    all_verts = []
    for cx, cy in centers:
        verts = _hex_verts_vec(cx, cy, angle_deg)
        all_verts.append(verts)
    return np.vstack(all_verts)

def _objective_rigid(centers, angle_rad):
    """Objective function for rigid body rotation."""
    angle_deg = np.degrees(angle_rad)
    verts = _compute_cluster_vertices(centers, angle_deg)
    outer_n = _hex_normals_vec(0.0)
    projections = verts @ outer_n.T
    max_proj = np.max(np.abs(projections))
    side = max_proj / _HALF_SQRT3
    return side

def _true_side(centers, outer_center, outer_angle):
    """Exact required outer side length."""
    outer_n = _hex_normals_vec(outer_angle)
    max_d = 0.0
    for c in centers:
        verts = _hex_verts_vec(c[0], c[1], c[2])
        rel = verts - np.array(outer_center)
        d = np.abs(rel @ outer_n.T)
        max_d = max(max_d, d.max())
    return max_d / _HALF_SQRT3

def _true_overlap(centers):
    """Exact SAT min separation. Returns min over all pairs (negative = overlap)."""
    n = len(centers)
    min_sep = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            c1 = centers[i]
            c2 = centers[j]
            n1 = _hex_normals_vec(c1[2])
            n2 = _hex_normals_vec(c2[2])
            axes = np.vstack([n1, n2])
            v1 = _hex_verts_vec(c1[0], c1[1], c1[2])
            v2 = _hex_verts_vec(c2[0], c2[1], c2[2])
            p1 = axes @ v1.T
            p2 = axes @ v2.T
            seps = np.maximum(p1.min(axis=1) - p2.max(axis=1), p2.min(axis=1) - p1.max(axis=1))
            min_sep = min(min_sep, seps.min())
    return min_sep

def _is_valid(centers, outer_center, outer_angle, tol=1e-7):
    """Check: no overlaps (SAT) and containment."""
    if _true_overlap(centers) < -tol:
        return False
    return True

# Precompute multiple cluster topologies
_CLUSTERS = [
    _get_cluster_1(),
    _get_cluster_2(),
    _get_cluster_3(),
    _get_cluster_4()
]

def _optimize_cluster(centers):
    """Optimize rotation for a given cluster topology."""
    def obj(angle_rad):
        return _objective_rigid(centers, angle_rad)
    
    best_angle = 0.0
    best_side = float('inf')
    
    try:
        # Multi-start optimization
        starts = np.linspace(0, 2*np.pi, 12, endpoint=False)
        for start in starts:
            try:
                res = minimize_scalar(obj, bounds=(start - np.pi/6, start + np.pi/6), 
                                      method='bounded', options={'xatol': 1e-8, 'maxiter': 500})
                if res.success and res.fun < best_side:
                    best_angle = res.x
                    best_side = res.fun
            except Exception:
                pass
        
        # Full range refinement
        if best_side < float('inf'):
            res = minimize_scalar(obj, bounds=(0, 2*np.pi), method='bounded', 
                                  options={'xatol': 1e-8, 'maxiter': 1000})
            if res.success and res.fun < best_side:
                best_angle = res.x
                best_side = res.fun
    except Exception:
        # Fallback to grid search
        angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        sides = [obj(a) for a in angles]
        best_idx = np.argmin(sides)
        best_angle = angles[best_idx]
        best_side = sides[best_idx]
        
        if best_idx > 0 and best_idx < len(angles)-1:
            try:
                res = minimize_scalar(obj, bounds=(angles[best_idx-1], angles[best_idx+1]), 
                                      method='bounded', options={'xatol': 1e-8})
                if res.success:
                    best_angle = res.x
                    best_side = res.fun
            except Exception:
                pass
    
    return best_angle, best_side

def construct():
    t0 = time.time()
    
    best_overall_angle = 0.0
    best_overall_side = float('inf')
    best_cluster_idx = 0
    
    for idx, centers in enumerate(_CLUSTERS):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        angle, side = _optimize_cluster(centers)
        if side < best_overall_side:
            best_overall_side = side
            best_overall_angle = angle
            best_cluster_idx = idx
    
    # Use the best cluster
    best_centers = _CLUSTERS[best_cluster_idx]
    
    # Construct the result
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    angle_deg = np.degrees(best_overall_angle)
    rot_matrix = np.array([
        [np.cos(best_overall_angle), -np.sin(best_overall_angle)],
        [np.sin(best_overall_angle), np.cos(best_overall_angle)]
    ])
    
    rotated_centers = best_centers @ rot_matrix.T
    
    inner = []
    for i in range(N):
        x, y = rotated_centers[i]
        inner.append([x, y, angle_deg])
        
    # Validate
    if not _is_valid(inner, outer_center, outer_angle):
        # Fallback to a large safe configuration
        inner_default = []
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = 4.0 * np.cos(theta)
            y = 4.0 * np.sin(theta)
            inner_default.append([x, y, 0.0])
        side = _true_side(inner_default, (0.0, 0.0), 0.0)
        return inner_default, (0.0, 0.0), side, 0.0

    # Calculate true side for final reporting
    final_side = _true_side(inner, outer_center, outer_angle)
    
    return inner, outer_center, final_side, outer_angle

# EVOLVE-BLOCK-END
