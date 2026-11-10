# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _get_all_vertices(positions, angles, side=1.0):
    """positions: (N,2), angles: (N,) in degrees. Returns (N*6, 2)."""
    verts = []
    for i in range(len(positions)):
        verts.append(_hex_vertices(positions[i,0], positions[i,1], side, angles[i]))
    return np.vstack(verts)

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    """
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _separation_distance(v1, v2):
    """
    Compute the minimum separation distance between two convex polygons (hexagons).
    Returns a positive value if separated, negative if overlapping (penetration depth approximation).
    We use SAT: find the minimum overlap across all axes.
    If max overlap is negative, they are separated by -max_overlap.
    """
    # v1, v2 are (6, 2) arrays of vertices
    
    # Generate normals for v1 and v2
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    
    normals = np.array(normals) # (12, 2)
    
    # Project v1 and v2 onto all normals
    proj1 = v1 @ normals.T # (6, 12)
    proj2 = v2 @ normals.T # (6, 12)
    
    min1 = np.min(proj1, axis=0)
    max1 = np.max(proj1, axis=0)
    min2 = np.min(proj2, axis=0)
    max2 = np.max(proj2, axis=0)
    
    # Overlap on each axis: min(max1, max2) - max(min1, min2)
    overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
    
    # If any overlap is < 0, they are separated.
    # The "distance" is the maximum negative overlap (closest to 0 from below).
    # If all overlaps are > 0, they are overlapping.
    # We define separation as -max(overlaps).
    # If max(overlaps) < 0, separation = -max(overlaps) > 0.
    # If max(overlaps) > 0, separation = -max(overlaps) < 0 (penetration).
    
    return -np.max(overlaps)

def _check_separation(pos1, ang1, pos2, ang2, side=1.0, tol=1e-6):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated (no overlap), False if overlapping.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
    
    # Get all edge normals from both hexagons (12 axes)
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    
    normals = np.array(normals)
    
    proj1 = v1 @ normals.T
    proj2 = v2 @ normals.T
    
    min1 = np.min(proj1, axis=0)
    max1 = np.max(proj1, axis=0)
    min2 = np.min(proj2, axis=0)
    max2 = np.max(proj2, axis=0)
    
    # Separation condition: intervals do not overlap
    # max1 < min2 - tol OR max2 < min1 - tol
    # Equivalent to: max1 - min2 < -tol OR max2 - min1 < -tol
    # Or: min(max1 - min2, max2 - min1) < -tol
    
    # Let's just check if any axis has a gap
    # Gap on axis i = min(max1[i] - min2[i], max2[i] - min1[i])
    # If gap < -tol, they are separated.
    
    # Actually, standard SAT check for separation:
    # For each axis, if max1 < min2 - tol or max2 < min1 - tol, return True (separated)
    for i in range(len(normals)):
        if max1[i] < min2[i] - tol or max2[i] < min1[i] - tol:
            return True
            
    return False

def _is_valid_config(pos, ang, side=1.0, tol=1e-6):
    """
    Check if the configuration is valid:
    1. All inner hexagons are separated.
    2. All inner hexagons are inside the outer hexagon (computed from pos, ang).
    """
    n = len(pos)
    
    # 1. Check separation
    for i in range(n):
        for j in range(i+1, n):
            if not _check_separation(pos[i], ang[i], pos[j], ang[j], side, tol):
                return False, np.inf # Invalid, infinite cost
        
    # 2. Check containment
    vertices = _get_all_vertices(pos, ang, side)
    s = _min_outer_side(vertices)
    
    # Check if vertices are actually inside the hexagon of side s
    # The _min_outer_side computes s such that they are inside.
    # But we need to ensure the computed s is consistent with the geometry.
    # Actually, _min_outer_side returns the MINIMUM s. If we use that s, they are inside.
    # So validity is just separation.
    
    return True, s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    n = 12
    side = 1.0
    
    # Initial configuration: Hexagonal grid
    init_positions = []
    init_angles = []
    for row in range(4):
        for col in range(3):
            x = col * 3.0
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 1.5
            init_positions.append([x, y])
            init_angles.append(0.0)
    
    init_positions = np.array(init_positions)
    init_angles = np.array(init_angles)
    
    # Center
    centroid = np.mean(init_positions, axis=0)
    init_positions -= centroid
    
    best_pos = init_positions.copy()
    best_ang = init_angles.copy()
    _, best_s = _is_valid_config(best_pos, best_ang, side)
    
    # If initial is invalid (overlap), we need to fix it or use a penalty
    # The grid is valid, so best_s is finite.
    
    # Simulated Annealing / Stochastic Local Search
    current_pos = best_pos.copy()
    current_ang = best_ang.copy()
    _, current_s = _is_valid_config(current_pos, current_ang, side)
    
    # Objective: Minimize s. If invalid, add large penalty.
    def objective(pos, ang):
        valid, s = _is_valid_config(pos, ang, side)
        if not valid:
            # Calculate penetration depth penalty to guide search
            # Simple penalty: number of overlapping pairs
            pen = 0
            for i in range(n):
                for j in range(i+1, n):
                    if not _check_separation(pos[i], ang[i], pos[j], ang[j], side, 1e-6):
                        pen += 1
            return s + 10.0 * pen
        return s
    
    current_obj = objective(current_pos, current_ang)
    best_obj = current_obj
    
    T = 0.5  # Initial temperature
    T_min = 1e-6
    alpha = 0.995
    
    steps = 0
    while time.time() < deadline:
        # Generate a random move
        move_type = random.random()
        
        new_pos = current_pos.copy()
        new_ang = current_ang.copy()
        
        if move_type < 0.6:
            # Move one hexagon
            idx = random.randint(0, n-1)
            dx = random.gauss(0, 0.2)
            dy = random.gauss(0, 0.2)
            new_pos[idx] += [dx, dy]
        elif move_type < 0.8:
            # Rotate one hexagon
            idx = random.randint(0, n-1)
            da = random.gauss(0, 10.0)
            new_ang[idx] += da
        elif move_type < 0.9:
            # Move two hexagons in opposite directions (preserve center of mass)
            i1 = random.randint(0, n-1)
            i2 = random.randint(0, n-1)
            if i1 == i2:
                continue
            dx = random.gauss(0, 0.2)
            dy = random.gauss(0, 0.2)
            new_pos[i1] += [dx, dy]
            new_pos[i2] -= [dx, dy]
        else:
            # Global rotation (might help with symmetry breaking or alignment)
            # Actually, rotating all hexagons together doesn't change relative positions,
            # but changes orientation relative to the outer hexagon.
            # This can be useful if the outer hexagon orientation matters (it doesn't, it's free).
            # Wait, the outer hexagon is computed to fit the inner ones.
            # So global rotation of inner hexagons just rotates the whole cluster.
            # The minimal enclosing hexagon side s is invariant under global rotation?
            # No, the enclosing hexagon is fixed in orientation (angle 0).
            # So rotating the inner cluster changes which vertices hit the outer edges.
            # This IS a useful move.
            da = random.gauss(0, 5.0)
            # Rotate all positions around origin
            theta = np.radians(da)
            c, s_ = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s_], [s_, c]])
            new_pos = (rot @ new_pos.T).T
            # Rotate all angles
            new_ang = new_ang + da
        
        # Evaluate new state
        new_obj = objective(new_pos, new_ang)
        
        # Acceptance criterion
        if new_obj < current_obj:
            current_pos = new_pos
            current_ang = new_ang
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_pos = current_pos.copy()
                best_ang = current_ang.copy()
        else:
            # Metropolis criterion
            delta = new_obj - current_obj
            if T > T_min:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_pos = new_pos
                    current_ang = new_ang
                    current_obj = new_obj
        
        T *= alpha
        steps += 1
        
        # Restart if stuck for too long?
        # Simple: if T is very low, reset T occasionally to explore new regions
        if steps % 1000 == 0 and T < 0.01:
            T = 0.1
            # Perturb current best to escape local minima
            current_pos = best_pos + random.normal(0, 0.5, size=(n, 2))
            current_ang = best_ang + random.normal(0, 10, size=n)
            current_obj = objective(current_pos, current_ang)
            
    # Final validation
    valid, final_s = _is_valid_config(best_pos, best_ang, side)
    if not valid:
        # Fallback to a known valid config if best is invalid (shouldn't happen if we track valid bests)
        # But objective function allows invalid states with penalty.
        # We must return a valid config.
        # Let's ensure best_pos is valid.
        # If best_pos is invalid, we need to find a valid one.
        # The initial grid was valid.
        valid_init, s_init = _is_valid_config(init_positions, init_angles, side)
        if valid_init:
            best_pos = init_positions
            best_ang = init_angles
            final_s = s_init
    
    inner_hex_data = np.column_stack([best_pos, best_ang])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, final_s

# EVOLVE-BLOCK-END
