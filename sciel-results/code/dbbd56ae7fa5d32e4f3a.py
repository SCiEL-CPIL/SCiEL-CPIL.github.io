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
    center = np.mean(verts, axis=0)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def min_gap_between_hexes(v1, v2):
    """
    Compute the separation gap between two hexagons using their 12 edge normals.
    Returns the maximum gap over all normals (negative means overlap).
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
        
        gap1 = min2 - max1
        gap2 = min1 - max2
        gap = max(gap1, gap2)
        if gap > max_gap:
            max_gap = gap
    
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


def is_valid_configuration(inner_hex_data, outer_angle_deg, tol=1e-6):
    """Check if the configuration is valid (all hexagons separated)."""
    for i in range(11):
        for j in range(i + 1, 11):
            if not check_separation(inner_hex_data[i], inner_hex_data[j], tol):
                return False
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = 620.0
    safety_margin = 10.0
    
    h = math.sqrt(3) / 2  # height of a unit hexagon row
    
    # Start with the 4-3-4 honeycomb arrangement
    coords = []
    for i in range(4):
        coords.append([i * 3.0, 0.0])
    for i in range(3):
        coords.append([1.5 + i * 3.0, 2 * h])
    for i in range(4):
        coords.append([i * 3.0, 4 * h])
    
    # Center the configuration
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    
    # Slight expansion to ensure separation
    scale = 1.0002
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
    best_inner = inner_hex_data.copy()
    
    # Find the best outer angle with the initial configuration
    for outer_angle_deg in np.arange(0, 30.01, 0.5):
        if time.time() - start_time > deadline - safety_margin:
            break
        side = calculate_min_side(inner_hex_data, outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg
            best_inner = inner_hex_data.copy()
    
    # Refine the outer angle
    if time.time() - start_time < deadline - safety_margin:
        for outer_angle_deg in np.arange(max(0, best_outer_angle - 1.0), min(30, best_outer_angle + 1.0) + 0.01, 0.05):
            if time.time() - start_time > deadline - safety_margin:
                break
            side = calculate_min_side(inner_hex_data, outer_angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_angle_deg
                best_inner = inner_hex_data.copy()
    
    # Local optimization using gradient-free search
    if time.time() - start_time < deadline - safety_margin:
        current_pos = best_inner.copy()
        current_best = best_side
        
        # Multi-scale local search
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        for step in step_sizes:
            if time.time() - start_time > deadline - safety_margin:
                break
            improved = True
            while improved and time.time() - start_time < deadline - safety_margin:
                improved = False
                for i in range(11):
                    if time.time() - start_time > deadline - safety_margin:
                        break
                    # Try 8 directions
                    for dx in [-step, -step * 0.707, 0, step * 0.707, step, step * 0.707, 0, -step * 0.707]:
                        for dy in [0, step, step * 0.707, 0, -step, -step * 0.707, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
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
                            if new_side < current_best - 1e-12:
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
            best_inner = current_pos
    
    # Try different initial configurations
    if time.time() - start_time < deadline - safety_margin:
        # Try a 3-5-3 arrangement
        coords2 = []
        for i in range(3):
            coords2.append([i * 3.0, 0.0])
        for i in range(5):
            coords2.append([i * 3.0, 2 * h])
        for i in range(3):
            coords2.append([i * 3.0, 4 * h])
        
        cx2 = np.mean([c[0] for c in coords2])
        cy2 = np.mean([c[1] for c in coords2])
        for c in coords2:
            c[0] -= cx2
            c[1] -= cy2
        
        scale2 = 1.0002
        for c in coords2:
            c[0] *= scale2
            c[1] *= scale2
        
        inner2 = np.zeros((11, 3))
        for i, c in enumerate(coords2):
            inner2[i, 0] = c[0]
            inner2[i, 1] = c[1]
            inner2[i, 2] = 0.0
        
        for outer_angle_deg in np.arange(0, 30.01, 1.0):
            if time.time() - start_time > deadline - safety_margin:
                break
            side = calculate_min_side(inner2, outer_angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_angle_deg
                best_inner = inner2.copy()
        
        # Local refinement for this configuration
        if time.time() - start_time < deadline - safety_margin:
            current_pos = best_inner.copy()
            current_best = best_side
            step_sizes2 = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
            for step in step_sizes2:
                if time.time() - start_time > deadline - safety_margin:
                    break
                improved = True
                while improved and time.time() - start_time < deadline - safety_margin:
                    improved = False
                    for i in range(11):
                        if time.time() - start_time > deadline - safety_margin:
                            break
                        for dx in [-step, 0, step]:
                            for dy in [-step, 0, step]:
                                if dx == 0 and dy == 0:
                                    continue
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
                                if new_side < current_best - 1e-12:
                                    current_pos = test_pos
                                    current_best = new_side
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if current_best < best_side:
                best_side = current_best
                best_inner = current_pos
    
    # Try a circular-ish arrangement
    if time.time() - start_time < deadline - safety_margin:
        # One in center, 6 around it, 4 more
        coords3 = [[0.0, 0.0]]
        for i in range(6):
            angle = i * 60
            x = 3.0 * math.cos(math.radians(angle))
            y = 3.0 * math.sin(math.radians(angle))
            coords3.append([x, y])
        # Add 4 more at larger radius
        for i in range(4):
            angle = 30 + i * 90
            x = 5.5 * math.cos(math.radians(angle))
            y = 5.5 * math.sin(math.radians(angle))
            coords3.append([x, y])
        
        cx3 = np.mean([c[0] for c in coords3])
        cy3 = np.mean([c[1] for c in coords3])
        for c in coords3:
            c[0] -= cx3
            c[1] -= cy3
        
        scale3 = 1.0002
        for c in coords3:
            c[0] *= scale3
            c[1] *= scale3
        
        inner3 = np.zeros((11, 3))
        for i, c in enumerate(coords3):
            inner3[i, 0] = c[0]
            inner3[i, 1] = c[1]
            inner3[i, 2] = 0.0
        
        for outer_angle_deg in np.arange(0, 30.01, 1.0):
            if time.time() - start_time > deadline - safety_margin:
                break
            side = calculate_min_side(inner3, outer_angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_angle_deg
                best_inner = inner3.copy()
        
        # Local refinement
        if time.time() - start_time < deadline - safety_margin:
            current_pos = best_inner.copy()
            current_best = best_side
            step_sizes3 = [0.1, 0.05, 0.02, 0.01, 0.005]
            for step in step_sizes3:
                if time.time() - start_time > deadline - safety_margin:
                    break
                improved = True
                while improved and time.time() - start_time < deadline - safety_margin:
                    improved = False
                    for i in range(11):
                        if time.time() - start_time > deadline - safety_margin:
                            break
                        for dx in [-step, 0, step]:
                            for dy in [-step, 0, step]:
                                if dx == 0 and dy == 0:
                                    continue
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
                                if new_side < current_best - 1e-12:
                                    current_pos = test_pos
                                    current_best = new_side
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if current_best < best_side:
                best_side = current_best
                best_inner = current_pos
    
    # Final refinement of the best configuration found
    if time.time() - start_time < deadline - safety_margin:
        current_pos = best_inner.copy()
        current_best = best_side
        
        # Fine-grained search
        for step in [0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() - start_time > deadline - safety_margin:
                break
            improved = True
            while improved and time.time() - start_time < deadline - safety_margin:
                improved = False
                for i in range(11):
                    if time.time() - start_time > deadline - safety_margin:
                        break
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
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
                            if new_side < current_best - 1e-12:
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
            best_inner = current_pos
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return best_inner, outer_hex_data, best_side


# EVOLVE-BLOCK-END
