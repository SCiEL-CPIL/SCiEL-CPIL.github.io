# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, angle angle_deg."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def get_edge_normals(verts):
    """Get outward-pointing edge normals for a hexagon given its vertices."""
    normals = []
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        center = np.mean(verts, axis=0)
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def min_gap_between_hexes(v1, v2, tol=1e-9):
    """
    Compute the minimum gap between two hexagons using their 12 edge normals.
    Returns the maximum over all normals of the gap (negative means overlap).
    """
    n1 = get_edge_normals(v1)
    n2 = get_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    max_gap = -np.inf
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap = max(min2 - max1, min1 - max2)
        max_gap = max(max_gap, gap)
    
    return max_gap


def check_separation(hex1, hex2, tol=1e-6):
    """
    Check if two hexagons are separated: on at least one of their 12 edge normals,
    the projected intervals are apart by more than tol.
    """
    x1, y1, a1 = hex1
    x2, y2, a2 = hex2
    
    v1 = hexagon_vertices(x1, y1, 1.0, a1)
    v2 = hexagon_vertices(x2, y2, 1.0, a2)
    
    gap = min_gap_between_hexes(v1, v2)
    return gap > tol


def calculate_min_side(inner_hex_data, outer_angle_deg):
    """
    Calculate the minimum side length S of an outer regular hexagon centered at origin
    with given angle that contains all inner hexagons.
    """
    normal_angles = [outer_angle_deg + 30 + 60 * k for k in range(6)]
    normals = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in normal_angles])
    
    all_vertices = []
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        verts = hexagon_vertices(x, y, 1.0, angle)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    max_projs = np.max(all_vertices @ normals.T, axis=0)
    max_proj = np.max(max_projs)
    side = 2.0 * max_proj / math.sqrt(3)
    
    return side


def objective(params, outer_angle_deg, penalty_weight=1000.0):
    """
    Objective function for optimization: minimize outer hex side length plus constraint penalties.
    params: array of shape (11, 2) containing (x, y) for each hexagon.
    """
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = params[2*i]
        inner_hex_data[i, 1] = params[2*i+1]
        inner_hex_data[i, 2] = 0.0
    
    side = calculate_min_side(inner_hex_data, outer_angle_deg)
    
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            v1 = hexagon_vertices(x1, y1, 1.0, a1)
            v2 = hexagon_vertices(x2, y2, 1.0, a2)
            gap = min_gap_between_hexes(v1, v2)
            if gap < 1e-6:
                penalty += (1e-6 - gap) ** 2
    
    return side + penalty_weight * penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0
    
    h = math.sqrt(3) / 2
    coords = []
    for i in range(4):
        coords.append([i * 3.0, 0.0])
    for i in range(3):
        coords.append([1.5 + i * 3.0, 2 * h])
    for i in range(4):
        coords.append([i * 3.0, 4 * h])
    
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    
    scale = 1.0001
    for c in coords:
        c[0] *= scale
        c[1] *= scale
    
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(coords):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0
    
    best_side = float('inf')
    best_outer_angle = 0.0
    best_positions = inner_hex_data.copy()
    
    for outer_angle_deg in np.arange(0, 30.01, 0.5):
        if time.time() - start_time > deadline - safety_margin:
            break
        side = calculate_min_side(inner_hex_data, outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg
    
    if time.time() - start_time < deadline - safety_margin:
        for outer_angle_deg in np.arange(max(0, best_outer_angle - 1.0), min(30, best_outer_angle + 1.0) + 0.01, 0.1):
            if time.time() - start_time > deadline - safety_margin:
                break
            side = calculate_min_side(inner_hex_data, outer_angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_angle_deg
    
    if time.time() - start_time < deadline - safety_margin:
        init_params = np.zeros(22)
        for i in range(11):
            init_params[2*i] = inner_hex_data[i, 0]
            init_params[2*i+1] = inner_hex_data[i, 1]
        
        best_opt_side = best_side
        best_opt_params = init_params.copy()
        best_opt_angle = best_outer_angle
        
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() - start_time > deadline - safety_margin:
                break
            
            if r == 0:
                x0 = init_params.copy()
            else:
                np.random.seed(42 + r)
                x0 = init_params + np.random.randn(22) * 0.1
            
            try:
                result = minimize(
                    objective,
                    x0,
                    args=(best_opt_angle,),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if time.time() - start_time > deadline - safety_margin:
                    break
                
                test_hex_data = np.zeros((11, 3))
                for i in range(11):
                    test_hex_data[i, 0] = result.x[2*i]
                    test_hex_data[i, 1] = result.x[2*i+1]
                    test_hex_data[i, 2] = 0.0
                
                valid = True
                for i in range(11):
                    for j in range(i+1, 11):
                        if not check_separation(test_hex_data[i], test_hex_data[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    test_side = calculate_min_side(test_hex_data, best_opt_angle)
                    if test_side < best_opt_side:
                        best_opt_side = test_side
                        best_opt_params = result.x.copy()
                        best_positions = test_hex_data.copy()
                        
                        if time.time() - start_time < deadline - safety_margin:
                            for oa in np.arange(max(0, best_opt_angle - 2.0), min(30, best_opt_angle + 2.0) + 0.01, 0.1):
                                if time.time() - start_time > deadline - safety_margin:
                                    break
                                s = calculate_min_side(test_hex_data, oa)
                                if s < best_opt_side:
                                    best_opt_side = s
                                    best_opt_angle = oa
            except Exception:
                continue
        
        best_side = best_opt_side
        best_outer_angle = best_opt_angle
        inner_hex_data = best_positions
    
    if time.time() - start_time < deadline - safety_margin:
        for oa in np.arange(max(0, best_outer_angle - 0.5), min(30, best_outer_angle + 0.5) + 0.001, 0.01):
            if time.time() - start_time > deadline - safety_margin:
                break
            s = calculate_min_side(inner_hex_data, oa)
            if s < best_side:
                best_side = s
                best_outer_angle = oa
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    outer_hex_side_length = best_side
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
