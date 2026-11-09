# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(verts):
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm > 1e-15:
            normals[i] = [nx / norm, ny / norm]
    return normals

def _separation_check(verts1, verts2, tol=1e-6):
    normals1 = _hex_edge_normals(verts1)
    normals2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([normals1, normals2])
    for normal in all_normals:
        proj1 = verts1 @ normal
        proj2 = verts2 @ normal
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False

def _is_inside_outer(verts_inner, outer_verts, tol=1e-6):
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        for v in verts_inner:
            cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
            if cross < -tol:
                return False
    return True

def _validate(inner_data, outer_data, s):
    n_inner = len(inner_data)
    if n_inner != 12:
        return False
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    inner_verts_list = []
    for i in range(n_inner):
        x, y, ang = inner_data[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False
    
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            if not _separation_check(inner_verts_list[i], inner_verts_list[j]):
                return False
    
    for i in range(n_inner):
        if not _is_inside_outer(inner_verts_list[i], outer_verts):
            return False
    
    return True

def _compute_min_outer(inner_data, outer_center, outer_angle):
    lo = 1.0
    hi = 50.0
    for _ in range(20):
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], hi, outer_angle)
        ok = True
        for i in range(len(inner_data)):
            x, y, ang = inner_data[i]
            verts = _hex_vertices(x, y, 1.0, ang)
            if not _is_inside_outer(verts, outer_verts):
                ok = False
                break
        if ok:
            break
        hi *= 2
    if not ok:
        return hi
    
    for _ in range(60):
        mid = (lo + hi) / 2
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], mid, outer_angle)
        ok = True
        for i in range(len(inner_data)):
            x, y, ang = inner_data[i]
            verts = _hex_vertices(x, y, 1.0, ang)
            if not _is_inside_outer(verts, outer_verts):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi

def _objective_symmetric(params):
    r1, r2, theta = params
    # r1: radius of inner ring (6 hexes)
    # r2: radius of outer ring (6 hexes)
    # theta: rotation of inner hexagons
    
    centers = np.zeros((12, 2))
    angles = np.zeros(12)
    
    # Inner ring: 6 hexes at angles 0, 60, ...
    for k in range(6):
        ang_center = np.radians(60 * k)
        centers[2*k] = r1 * np.array([np.cos(ang_center), np.sin(ang_center)])
        angles[2*k] = theta
    
    # Outer ring: 6 hexes at angles 30, 90, ...
    for k in range(6):
        ang_center = np.radians(60 * k + 30)
        centers[2*k + 1] = r2 * np.array([np.cos(ang_center), np.sin(ang_center)])
        angles[2*k + 1] = theta
    
    inner_data = np.column_stack([centers, angles])
    
    # Check separation
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(verts)
        
    for i in range(12):
        for j in range(i+1, 12):
            if not _separation_check(inner_verts_list[i], inner_verts_list[j]):
                return 1000.0 # Penalty for overlap
    
    # Compute min s
    # Outer hexagon centered at 0, angle 0
    # Normals of outer hexagon (angle 0): vertices at 0, 60...
    # Edge normals are perpendicular to edges.
    # Vertices: (s, 0), (s/2, s*sqrt(3)/2)...
    # Edge 0: from (s,0) to (s/2, s*sqrt(3)/2).
    # Vector: (-s/2, s*sqrt(3)/2). Normal: (s*sqrt(3)/2, s/2) -> (sqrt(3), 1) normalized.
    # Actually, simpler: The hexagon is intersection of half planes n_k . x <= A.
    # For angle 0 hexagon, the normals are at 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0
    for verts in inner_verts_list:
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    # A = max_proj
    # s * sqrt(3) / 2 = A  => s = 2 * A / sqrt(3)
    s = 2 * max_proj / np.sqrt(3)
    
    if s < np.sqrt(12) - 1e-9:
        return 1000.0
        
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    best_s = np.inf
    best_params = None
    
    # Initial guesses
    # r1 ~ 1.7 (touching), r2 ~ 3.0
    starts = [
        [1.732, 3.0, 0],
        [1.732, 3.0, 30],
        [1.5, 3.0, 0],
        [2.0, 3.5, 0],
        [1.732, 3.2, 15],
        [1.8, 3.1, 0],
    ]
    
    for start in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _objective_symmetric,
                start,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_s and res.fun < 100: # 100 is a safe upper bound for valid s
                best_s = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to a known valid configuration
        # Just use a large hexagon
        inner_data = np.zeros((12, 3))
        for i in range(12):
            inner_data[i, 0] = (i % 4) * 2 - 3
            inner_data[i, 1] = (i // 4) * 2 - 1
            inner_data[i, 2] = 0
        outer_data = np.array([0, 0, 0])
        s = 10.0
        return inner_data, outer_data, s
        
    r1, r2, theta = best_params
    
    centers = np.zeros((12, 2))
    angles = np.zeros(12)
    for k in range(6):
        ang_center = np.radians(60 * k)
        centers[2*k] = r1 * np.array([np.cos(ang_center), np.sin(ang_center)])
        angles[2*k] = theta
    for k in range(6):
        ang_center = np.radians(60 * k + 30)
        centers[2*k + 1] = r2 * np.array([np.cos(ang_center), np.sin(ang_center)])
        angles[2*k + 1] = theta
        
    inner_data = np.column_stack([centers, angles])
    outer_data = np.array([0, 0, 0])
    
    # Final validation and s calculation
    s = _compute_min_outer(inner_data, outer_data, 0)
    
    if _validate(inner_data, outer_data, s):
        return inner_data, outer_data, s
    else:
        # If validation fails, return a very large valid box
        inner_data = np.zeros((12, 3))
        for i in range(12):
            inner_data[i, 0] = (i % 4) * 2 - 3
            inner_data[i, 1] = (i // 4) * 2 - 1
            inner_data[i, 2] = 0
        outer_data = np.array([0, 0, 0])
        s = 10.0
        return inner_data, outer_data, s

# EVOLVE-BLOCK-END
