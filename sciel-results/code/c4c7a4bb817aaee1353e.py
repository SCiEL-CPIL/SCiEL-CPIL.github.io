# EVOLVE-BLOCK-START
import numpy as np
import math
import time


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


def check_separation(hex1, hex2, tol=1e-6):
    """
    Check if two hexagons are separated: on at least one of their 12 edge normals,
    the projected intervals are apart by more than tol.
    """
    x1, y1, a1 = hex1
    x2, y2, a2 = hex2
    
    v1 = hexagon_vertices(x1, y1, 1.0, a1)
    v2 = hexagon_vertices(x2, y2, 1.0, a2)
    
    n1 = get_edge_normals(v1)
    n2 = get_edge_normals(v2)
    
    all_normals = np.vstack([n1, n2])
    
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False


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


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 5.0
    
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
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    outer_hex_side_length = best_side
    
    if time.time() - start_time < deadline - safety_margin:
        current_best = best_side
        current_pos = inner_hex_data.copy()
        
        step_sizes = [0.05, 0.01, 0.001]
        for step in step_sizes:
            if time.time() - start_time > deadline - safety_margin:
                break
            improved = True
            while improved and time.time() - start_time < deadline - safety_margin:
                improved = False
                for i in range(11):
                    if time.time() - start_time > deadline - safety_margin:
                        break
                    for dx in [-step, step]:
                        for dy in [-step, step]:
                            test_pos = current_pos.copy()
                            test_pos[i, 0] += dx
                            test_pos[i, 1] += dy
                            
                            valid = True
                            for j in range(11):
                                if i == j:
                                    continue
                                if not check_separation(test_pos[i], test_pos[j], tol=1e-6):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side = calculate_min_side(test_pos, best_outer_angle)
                            if new_side < current_best - 1e-10:
                                current_pos = test_pos
                                current_best = new_side
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        if current_best < best_side:
            best_side = current_best
            inner_hex_data = current_pos
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
