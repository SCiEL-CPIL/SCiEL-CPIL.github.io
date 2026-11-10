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
        
        gap1 = min2 - max1  # positive if v2 is to the right of v1
        gap2 = min1 - max2  # positive if v1 is to the right of v2
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


def objective(params, outer_angle_deg):
    """
    Objective function for optimization: returns the outer hexagon side length
    plus a penalty for any violations of separation constraints.
    """
    # params: 22 values (11 hexagons * 2 coords)
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = params[2*i]
        inner_hex_data[i, 1] = params[2*i+1]
        inner_hex_data[i, 2] = 0.0
    
    # Compute the outer hexagon side
    side = calculate_min_side(inner_hex_data, outer_angle_deg)
    
    # Check separation constraints
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            if not check_separation(inner_hex_data[i], inner_hex_data[j], tol=1e-6):
                # Compute the actual gap
                v1 = hexagon_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, 0.0)
                v2 = hexagon_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, 0.0)
                gap = min_gap_between_hexes(v1, v2)
                if gap < 1e-6:
                    penalty += (1e-6 - gap) ** 2
    
    return side + 100 * penalty


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
    best_inner = inner_hex_data.copy()
    
    # First, find the best outer angle with the initial configuration
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
    
    # Now use scipy minimize to optimize the positions
    if time.time() - start_time < deadline - safety_margin:
        x0 = best_inner[:, :2].flatten()
        
        # Try different optimization methods and initial perturbations
        for method in ['L-BFGS-B', 'Nelder-Mead']:
            if time.time() - start_time > deadline - safety_margin:
                break
            
            # Try a few different initial conditions
            for trial in range(3):
                if time.time() - start_time > deadline - safety_margin:
                    break
                
                if trial == 0:
                    x_init = x0.copy()
                else:
                    # Add small random perturbations
                    np.random.seed(trial * 1234 + 5678)
                    x_init = x0 + np.random.randn(22) * 0.01
                
                try:
                    result = minimize(
                        objective,
                        x_init,
                        args=(best_outer_angle,),
                        method=method,
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    
                    if time.time() - start_time > deadline - safety_margin:
                        break
                    
                    # Check if the result is valid
                    test_inner = np.zeros((11, 3))
                    for i in range(11):
                        test_inner[i, 0] = result.x[2*i]
                        test_inner[i, 1] = result.x[2*i+1]
                        test_inner[i, 2] = 0.0
                    
                    # Verify separation
                    valid = True
                    for i in range(11):
                        for j in range(i+1, 11):
                            if not check_separation(test_inner[i], test_inner[j], tol=1e-6):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        new_side = calculate_min_side(test_inner, best_outer_angle)
                        if new_side < best_side:
                            best_side = new_side
                            best_inner = test_inner
                            
                except Exception:
                    pass
    
    # Final local refinement with small steps
    if time.time() - start_time < deadline - safety_margin:
        current_best = best_side
        current_pos = best_inner.copy()
        
        step_sizes = [0.01, 0.001, 0.0001]
        for step in step_sizes:
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
            best_inner = current_pos
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return best_inner, outer_hex_data, best_side


# EVOLVE-BLOCK-END
