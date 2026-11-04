import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(theta=0.0):
    """Return 6 vertices of a unit regular hexagon rotated by theta."""
    angles = theta + np.arange(6) * (math.pi / 3)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_outer_side(centers, angles, outer_center, outer_angle):
    """
    Compute the minimum side length S of a regular hexagon (centered at outer_center,
    rotated by outer_angle) that contains all inner hexagons.
    
    A regular hexagon of side S contains a point p iff for all 6 outward normals,
    the projection of (p - outer_center) onto the normal is <= S * sqrt(3)/2.
    So S = max over all vertices of all inner hexagons, and over all 6 normals,
    of the projection / (sqrt(3)/2).
    """
    # All vertices of all inner hexagons
    all_verts = []
    for i in range(len(centers)):
        verts = _hex_vertices(angles[i])
        all_verts.append(centers[i] + verts)
    all_verts = np.vstack(all_verts)  # shape (6*N, 2)
    
    # Outer hexagon normals
    outer_normals = _hex_vertices(outer_angle)  # 6 unit normals
    
    # Relative to outer center
    rel = all_verts - outer_center
    
    # Projections onto each normal
    # shape (6*N, 6)
    projs = rel @ outer_normals.T
    
    # For each normal, the max projection gives the constraint
    max_projs = np.max(np.abs(projs), axis=0)  # shape (6,)
    
    # S = max_proj / (sqrt(3)/2)
    S = np.max(max_projs) / (math.sqrt(3) / 2.0)
    return S

def _check_no_overlap(centers, angles, eps=1e-6):
    """
    Check that no two inner hexagons overlap.
    Two convex polygons overlap iff one vertex is inside the other or edges cross.
    For regular hexagons, a simpler check: the minimum distance between any vertex
    of hex i and any vertex of hex j should be >= 0 (non-overlapping means no vertex
    of one is strictly inside the other, but for safety we check vertex-in-polygon).
    
    Actually, for disjoint closed convex sets, we need strict separation.
    A practical check: for each pair, compute the minimum distance between their
    vertex sets and edge sets. If min distance > eps, they're disjoint.
    
    For efficiency, we use a separating axis theorem check.
    """
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            vi = centers[i] + _hex_vertices(angles[i])
            vj = centers[j] + _hex_vertices(angles[j])
            
            # SAT: check all 12 normals (6 from each hex)
            sep = False
            for k in range(6):
                # Normal of edge k of hex i
                edge_vec = vi[(k+1) % 6] - vi[k]
                normal = np.array([-edge_vec[1], edge_vec[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-15)
                
                proj_i = vi @ normal
                proj_j = vj @ normal
                if np.max(proj_i) < np.min(proj_j) - eps or np.max(proj_j) < np.min(proj_i) - eps:
                    sep = True
                    break
            if sep:
                continue
            
            for k in range(6):
                edge_vec = vj[(k+1) % 6] - vj[k]
                normal = np.array([-edge_vec[1], edge_vec[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-15)
                
                proj_i = vi @ normal
                proj_j = vj @ normal
                if np.max(proj_i) < np.min(proj_j) - eps or np.max(proj_j) < np.min(proj_i) - eps:
                    sep = True
                    break
            if not sep:
                return False
    return True

def _smooth_objective(v, outer_center, outer_angle, penalty_weight=1000.0):
    """
    Smooth objective: minimize outer side length + overlap penalty.
    v = [x0, y0, theta0, x1, y1, theta1, ..., x10, y10, theta10]
    """
    n = N
    centers = v[:3*n].reshape(n, 2)
    angles = v[3*n:4*n]
    
    # Overlap penalty using smooth approximation
    # For each pair, compute minimum distance via SAT, penalize if negative
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = centers[i] + _hex_vertices(angles[i])
            vj = centers[j] + _hex_vertices(angles[j])
            
            # Compute min separation via SAT
            min_sep = float('inf')
            for k in range(6):
                edge_vec = vi[(k+1) % 6] - vi[k]
                norm = np.linalg.norm(edge_vec)
                if norm < 1e-15:
                    continue
                normal = np.array([-edge_vec[1], edge_vec[0]]) / norm
                proj_i = vi @ normal
                proj_j = vj @ normal
                sep = np.min(proj_j) - np.max(proj_i)
                min_sep = min(min_sep, sep)
            
            for k in range(6):
                edge_vec = vj[(k+1) % 6] - vj[k]
                norm = np.linalg.norm(edge_vec)
                if norm < 1e-15:
                    continue
                normal = np.array([-edge_vec[1], edge_vec[0]]) / norm
                proj_i = vi @ normal
                proj_j = vj @ normal
                sep = np.min(proj_j) - np.max(proj_i)
                min_sep = min(min_sep, sep)
            
            # Smooth penalty: if min_sep < 0, penalize quadratically
            if min_sep < 0:
                penalty += (min_sep) ** 2
    
    if penalty > 1e-4:
        return penalty_weight + penalty
    
    S = _min_outer_side(centers, angles, outer_center, outer_angle)
    return S

def _initial_configuration(method='hex_lattice'):
    """Generate an initial configuration of N inner hexagons."""
    if method == 'hex_lattice':
        # Hexagonal lattice: center + rings
        d = math.sqrt(3)
        centers = [(0.0, 0.0)]
        # Ring 1
        for i in range(6):
            a = i * math.pi / 3
            centers.append((d * math.cos(a), d * math.sin(a)))
        # Ring 2
        for q in range(-2, 3):
            for r in range(-2, 3):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) == 2:
                    x = d * (q + r / 2.0)
                    y = d * (math.sqrt(3) / 2.0) * r
                    centers.append((x, y))
        centers = centers[:N]
        angles = np.zeros(N)
        return np.array(centers), angles
    elif method == 'circle':
        # Place on a circle
        R = 2.0
        centers = []
        angles = []
        for i in range(N):
            a = 2 * math.pi * i / N
            centers.append((R * math.cos(a), R * math.sin(a)))
            angles.append(a)
        return np.array(centers), np.array(angles)
    else:
        # Random
        centers = np.random.randn(N, 2) * 2.0
        angles = np.random.uniform(0, math.pi / 3, N)
        return centers, angles

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 8.0
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    best_S = float('inf')
    best_v = None
    best_valid = False
    
    # Try multiple initial configurations
    init_methods = ['hex_lattice', 'circle']
    np.random.seed(42)
    
    for method in init_methods:
        if time.time() > deadline - safety_margin:
            break
        
        centers, angles = _initial_configuration(method)
        
        # Also try with some random perturbations
        for trial in range(3):
            if time.time() > deadline - safety_margin:
                break
            
            if trial == 0:
                c = centers.copy()
                a = angles.copy()
            else:
                c = centers + np.random.randn(N, 2) * 0.1
                a = angles + np.random.uniform(-0.1, 0.1, N)
            
            v0 = np.concatenate([c.flatten(), a])
            
            try:
                res = minimize(
                    _smooth_objective,
                    v0,
                    args=(outer_center, outer_angle, 1000.0),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfev': 5000, 'ftol': 1e-8}
                )
                if res.fun < best_S and res.fun < 50.0:  # reasonable bound
                    best_S = res.fun
                    best_v = res.x.copy()
                    # Check validity
                    c_res = best_v[:3*N].reshape(N, 2)
                    a_res = best_v[3*N:4*N]
                    if _check_no_overlap(c_res, a_res, eps=1e-5):
                        best_valid = True
            except Exception:
                pass
    
    # If we have a valid solution, refine it
    if best_v is not None:
        if not best_valid:
            # Try to fix overlaps with a penalty-based refinement
            try:
                res = minimize(
                    _smooth_objective,
                    best_v,
                    args=(outer_center, outer_angle, 10000.0),
                    method='Powell',
                    options={'maxiter': 1000, 'maxfev': 10000, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < 50.0:
                    c_res = res.x[:3*N].reshape(N, 2)
                    a_res = res.x[3*N:4*N]
                    if _check_no_overlap(c_res, a_res, eps=1e-5):
                        best_S = _min_outer_side(c_res, a_res, outer_center, outer_angle)
                        best_v = res.x.copy()
                        best_valid = True
            except Exception:
                pass
    
    # Fallback: if no valid solution found, use a simple lattice
    if best_v is None or not best_valid:
        d = math.sqrt(3)
        centers = [(0.0, 0.0)]
        for i in range(6):
            a = i * math.pi / 3
            centers.append((d * math.cos(a), d * math.sin(a)))
        for q in range(-2, 3):
            for r in range(-2, 3):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) == 2:
                    x = d * (q + r / 2.0)
                    y = d * (math.sqrt(3) / 2.0) * r
                    centers.append((x, y))
        centers = centers[:N]
        angles = np.zeros(N)
        S = _min_outer_side(np.array(centers), angles, outer_center, outer_angle)
        best_S = S
        best_v = np.concatenate([np.array(centers).flatten(), angles])
        best_valid = True
    
    # Format output
    centers_final = best_v[:3*N].reshape(N, 2)
    angles_final = best_v[3*N:4*N]
    
    inner_rows = []
    for i in range(N):
        cx, cy = centers_final[i]
        angle_deg = math.degrees(angles_final[i]) % (math.pi / 3)  # Normalize to [0, 60)
        inner_rows.append([float(cx), float(cy), float(angle_deg)])
    
    center = (float(outer_center[0]), float(outer_center[1]))
    side = float(best_S)
    angle_deg = float(math.degrees(outer_angle))
    
    return inner_rows, center, side, angle_deg
# EVOLVE-BLOCK-END
