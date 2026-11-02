"""
Hexagon packing: 12 unit hexagons in a regular hexagon, minimize outer side length.
Strategy: Compact hexagonal lattice arrangements + Nelder-Mead refinement on individual positions/angles.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 148.0  # Safety margin before 150s kill

# Unit hexagon vertices (flat-top, side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon (pointing to edge midpoints)
# Edge midpoints are at angles 30, 90, 150, 210, 270, 330
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])


def get_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts


def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    verts = get_vertices(np.array(centers, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # For a regular hexagon, apothem = side * sqrt(3)/2
    # So side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)


def check_validity(centers, outer_center, outer_side, outer_angle, tol=1e-9):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    verts = get_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # Check pairwise separation using SAT
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            # Check all 12 edge normals (6 from each hexagon)
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # Check containment in outer hexagon
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
    
    return True


def generate_hex_lattice_arrangements():
    """Generate several compact hexagonal lattice arrangements of 12 hexagons."""
    arrangements = []
    
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3) / 2, 1.5])
    
    # Pattern 1: 3-4-3-2 (rows)
    pattern1 = [
        (0, 0),
        (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1),
        (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3)
    ]
    
    # Pattern 2: 2-3-4-3 (rows)
    pattern2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    # Pattern 3: 1-3-4-3-1
    pattern3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    
    # Pattern 4: 4-4-4 (three rows of 4)
    pattern4 = [
        (-1.5, 0), (-0.5, 0), (0.5, 0), (1.5, 0),
        (-1.0, 1.5), (0.0, 1.5), (1.0, 1.5), (2.0, 1.5),
        (-1.5, 3.0), (-0.5, 3.0), (0.5, 3.0), (1.5, 3.0)
    ]
    
    # Pattern 5: 2-4-4-2
    pattern5 = [
        (-0.5, 0), (0.5, 0),
        (-1.5, 1.5), (-0.5, 1.5), (0.5, 1.5), (1.5, 1.5),
        (-1.5, 3.0), (-0.5, 3.0), (0.5, 3.0), (1.5, 3.0),
        (-0.5, 4.5), (0.5, 4.5)
    ]
    
    # Pattern 6: Hexagonal number pattern (center + rings)
    pattern6 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2)
    ]
    
    # Pattern 7: 3-3-3-3 (four rows of 3, staggered)
    pattern7 = [
        (-1, 0), (0, 0), (1, 0),
        (-0.5, 1.5), (0.5, 1.5), (1.5, 1.5),
        (-1, 3.0), (0, 3.0), (1, 3.0),
        (-0.5, 4.5), (0.5, 4.5), (1.5, 4.5)
    ]
    
    # Pattern 8: 2-3-3-2-2
    pattern8 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8]:
        centers = []
        for (i, j) in pattern:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], 0.0])
        arrangements.append(centers)
    
    return arrangements


def objective_nelder_mead(params, outer_center, outer_angle):
    """
    Objective function for Nelder-Mead optimization.
    params: (3*N,) array of [x1, y1, a1, x2, y2, a2, ...]
    Returns: required outer side length + penalty for overlaps
    """
    n = len(params) // 3
    centers = params.reshape(n, 3)
    
    # Compute required side length
    side = required_outer_side(centers, outer_center, outer_angle)
    
    # Check for overlaps and add penalty
    verts = get_vertices(centers)
    penalty = 0.0
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            max_sep = np.max(sep)
            if max_sep < 0:
                penalty += (-max_sep) ** 2
    
    return side + 1000.0 * penalty


def optimize_with_nelder_mead(centers, outer_center, outer_angle, max_iter=500):
    """
    Use Nelder-Mead to optimize individual positions and angles.
    """
    n = len(centers)
    x0 = np.array(centers).flatten()
    
    result = minimize(
        objective_nelder_mead,
        x0,
        args=(outer_center, outer_angle),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    
    if result.success or result.fun < required_outer_side(centers, outer_center, outer_angle):
        optimized_centers = result.x.reshape(n, 3)
        return optimized_centers
    return np.array(centers)


def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Generate initial arrangements
    arrangements = generate_hex_lattice_arrangements()
    
    for arr_idx, centers in enumerate(arrangements):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Coarse rotation search
        best_side_coarse = float('inf')
        best_angle_coarse = 0.0
        angles = np.linspace(0, 60, 60)
        
        for rot_deg in angles:
            rotated_centers = [[c[0], c[1], c[2] + rot_deg] for c in centers]
            side = required_outer_side(rotated_centers, (0.0, 0.0), 0.0)
            
            if side < best_side_coarse:
                best_side_coarse = side
                best_angle_coarse = rot_deg
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Fine rotation search
        best_side_fine = float('inf')
        best_angle_fine = best_angle_coarse
        angles_fine = np.linspace(best_angle_coarse - 2.5, best_angle_coarse + 2.5, 50)
        
        for rot_deg in angles_fine:
            rotated_centers = [[c[0], c[1], c[2] + rot_deg] for c in centers]
            side = required_outer_side(rotated_centers, (0.0, 0.0), 0.0)
            
            if side < best_side_fine:
                best_side_fine = side
                best_angle_fine = rot_deg
        
        # Apply best rotation
        rotated_centers = [[c[0], c[1], c[2] + best_angle_fine] for c in centers]
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            # Skip Nelder-Mead if running out of time
            final_centers = rotated_centers
            final_side = required_outer_side(final_centers, (0.0, 0.0), 0.0)
            
            if check_validity(final_centers, (0.0, 0.0), final_side, 0.0):
                if final_side < best_score:
                    best_score = final_side
                    inner_list = [list(c) for c in final_centers]
                    best_result = (inner_list, (0.0, 0.0), final_side, 0.0)
            continue
        
        # Nelder-Mead refinement
        try:
            optimized_centers = optimize_with_nelder_mead(rotated_centers, (0.0, 0.0), 0.0, max_iter=300)
            final_side = required_outer_side(optimized_centers, (0.0, 0.0), 0.0)
            
            if check_validity(optimized_centers, (0.0, 0.0), final_side, 0.0):
                if final_side < best_score:
                    best_score = final_side
                    inner_list = [list(c) for c in optimized_centers]
                    best_result = (inner_list, (0.0, 0.0), final_side, 0.0)
        except Exception:
            # Fall back to rotated centers if Nelder-Mead fails
            final_side = required_outer_side(rotated_centers, (0.0, 0.0), 0.0)
            if check_validity(rotated_centers, (0.0, 0.0), final_side, 0.0):
                if final_side < best_score:
                    best_score = final_side
                    inner_list = [list(c) for c in rotated_centers]
                    best_result = (inner_list, (0.0, 0.0), best_score, 0.0)
    
    # If we still have time, try some random perturbations of the best arrangement
    elapsed = time.time() - start_time
    if best_result is not None and elapsed < DEADLINE - 5:
        base_centers = [list(c) for c in best_result[0]]
        
        np.random.seed(42)
        for trial in range(10):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 2:
                break
            
            perturbed = []
            for c in base_centers:
                dx = np.random.uniform(-0.05, 0.05)
                dy = np.random.uniform(-0.05, 0.05)
                da = np.random.uniform(-1, 1)
                perturbed.append([c[0] + dx, c[1] + dy, c[2] + da])
            
            side = required_outer_side(perturbed, (0.0, 0.0), 0.0)
            
            if side < best_score:
                if check_validity(perturbed, (0.0, 0.0), side, 0.0):
                    best_score = side
                    inner_list = [list(c) for c in perturbed]
                    best_result = (inner_list, (0.0, 0.0), best_score, 0.0)
    
    if best_result is not None:
        return best_result
    
    # Fallback: circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0


if __name__ == "__main__":
    result = construct()
    print(f"Best outer side length: {result[2]:.6f}")
    print(f"Number of inner hexagons: {len(result[0])}")
