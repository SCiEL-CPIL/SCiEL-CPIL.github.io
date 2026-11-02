import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s hard limit

def _get_unit_hex_verts():
    """Vertices of a unit regular hexagon centered at origin, angle 0."""
    angles = np.arange(6) * np.pi / 3
    return np.column_stack((np.cos(angles), np.sin(angles)))

UNIT_VERTS = _get_unit_hex_verts()

def _transform_hex_verts(positions, angles):
    """
    positions: (N, 2) array
    angles: (N,) array in radians
    Returns: (N, 6, 2) array of vertices
    """
    N = len(positions)
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        theta = angles[i]
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + positions[i]
    return verts

def _check_disjoint_fast(verts, tol=1e-9):
    """
    Check if all hexagons are disjoint using SAT (Separating Axis Theorem).
    verts: (N, 6, 2)
    Returns True if disjoint.
    """
    N = len(verts)
    for i in range(N):
        for j in range(i + 1, N):
            found_separating = False
            for k in range(6):
                v1 = verts[i, k]
                v2 = verts[i, (k + 1) % 6]
                edge = v2 - v1
                n = np.array([edge[1], -edge[0]])
                proj_i = np.dot(verts[i], n)
                proj_j = np.dot(verts[j], n)
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                if max_i < min_j - tol or max_j < min_i - tol:
                    found_separating = True
                    break
            if found_separating:
                continue
            
            for k in range(6):
                v1 = verts[j, k]
                v2 = verts[j, (k + 1) % 6]
                edge = v2 - v1
                n = np.array([edge[1], -edge[0]])
                proj_i = np.dot(verts[i], n)
                proj_j = np.dot(verts[j], n)
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                if max_i < min_j - tol or max_j < min_i - tol:
                    found_separating = True
                    break
            
            if not found_separating:
                return False
    return True

def _min_enclosing_hex_side(points, outer_angle_rad):
    """
    Find min side length of regular hexagon with given orientation containing all points.
    points: (M, 2) array
    outer_angle_rad: float
    Returns: (center_x, center_y, side_length) or None
    """
    normals = []
    for k in range(6):
        a = outer_angle_rad + math.pi / 6 + k * math.pi / 3
        normals.append((math.cos(a), math.sin(a)))
    normals = np.array(normals) # (6, 2)
    
    A = []
    b = []
    
    p_dot_n = points @ normals.T # (M, 6)
    
    for k in range(6):
        nx, ny = normals[k]
        min_val = np.min(p_dot_n[:, k])
        max_val = np.max(p_dot_n[:, k])
        
        # c . n_k - R <= min_val
        A.append([nx, ny, -1.0])
        b.append(min_val)
        
        # -c . n_k - R <= -max_val
        A.append([-nx, -ny, -1.0])
        b.append(-max_val)
        
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    A = np.array(A)
    b = np.array(b)
    c_obj = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, 
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R
            return cx, cy, side
        else:
            return None
    except:
        return None

def construct():
    global _START_TIME, _DEADLINE
    
    fallback_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    fallback_return = (fallback_inner, (0.0, 0.0), 40.0, 0.0)
    
    if time.time() > _DEADLINE - 10.0:
        return fallback_return

    sqrt3 = math.sqrt(3)
    u = np.array([sqrt3, 0.0])
    v = np.array([sqrt3 / 2, 1.5])
    
    grid_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            grid_points.append(p)
            
    grid_points.sort(key=lambda p: np.dot(p, p))
    init_pos = np.array(grid_points[:N])
    init_ang = np.zeros(N)
    
    verts = _transform_hex_verts(init_pos, init_ang)
    
    if not _check_disjoint_fast(verts):
        return fallback_return

    all_verts = verts.reshape(-1, 2)
    
    final_best_side = float('inf')
    final_center = (0.0, 0.0)
    final_outer_angle = 0.0
    
    for outer_angle_deg in np.arange(0, 30.1, 0.5):
        res = _min_enclosing_hex_side(all_verts, math.radians(outer_angle_deg))
        if res:
            side = res[2]
            if side < final_best_side:
                final_best_side = side
                final_center = (res[0], res[1])
                final_outer_angle = outer_angle_deg
                
    if final_best_side > 1e5:
        return fallback_return

    inner = []
    for i in range(N):
        inner.append([init_pos[i, 0], init_pos[i, 1], math.degrees(init_ang[i])])
        
    return inner, final_center, final_best_side, final_outer_angle
