import numpy as np
import math
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(center, angle_deg, side=1.0):
    """Return vertices of a regular hexagon (side=1) centered at `center`, rotated by angle_deg."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    R = np.array([[c, -s], [s, c]])
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    local = side * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    rotated = local @ R.T
    return rotated + center

def _outer_hex_containment(side, angle_deg, center=(0.0, 0.0)):
    """Return the 6 normals and offsets of the outer hexagon (side `side`), rotated by angle_deg."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    R = np.array([[c, -s], [s, c]])
    normals_local = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)
    ])
    normals = normals_local @ R.T
    # For a regular hexagon of side S, the apothem is S * sqrt(3) / 2
    apothem = side * math.sqrt(3) / 2.0
    offsets = np.full(6, apothem)
    return normals, offsets, center

def _max_distance_to_boundary(points, normals, offsets, center):
    """For each point, compute max over the 6 constraints of (n . (p - center) - offset).
    If all <= 0, the point is inside (or on boundary). Return max violation."""
    # points: (M, 2), normals: (6, 2), offsets: (6,), center: (2,)
    # For each point i and each normal j: n_j . (p_i - center) - offset_j
    # Shape: (M, 6)
    diff = points - center  # (M, 2)
    proj = diff @ normals.T  # (M, 6)
    violations = proj - offsets[None, :]  # (M, 6)
    return np.max(violations)

def _smooth_max(values, temperature=0.01):
    """Smooth maximum via log-sum-exp."""
    m = np.max(values)
    if m == -np.inf:
        return m
    return m + temperature * math.log(np.sum(np.exp((values - m) / temperature)))

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Initial configuration: 3x4 grid
    u = np.array([math.sqrt(3), 0.0])
    v = np.array([math.sqrt(3)/2.0, 1.5])
    
    centers = []
    angles = []
    for i in range(3):
        for j in range(4):
            c = i * u + j * v
            centers.append(c)
            angles.append(0.0)
    
    centers = np.array(centers)
    angles = np.array(angles)
    
    # Center the cluster
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Optimize outer rotation first
    best_side = float('inf')
    best_outer_angle = 0.0
    
    # Precompute all vertices for initial config
    all_vertices = np.vstack([_hex_vertices(centers[i], angles[i]) for i in range(N)])
    
    # Try different outer hexagon angles
    for phi_deg in np.arange(0, 60, 1.0):
        if time.time() > deadline - 60:
            break
        
        normals, offsets, outer_center = _outer_hex_containment(1.0, phi_deg, (0, 0))
        
        # For a hexagon of side S, the "distance" to boundary in direction n is S*sqrt(3)/2
        # We want to find min S such that max over all points and normals of (n . p - offset) <= 0
        # offset = S * sqrt(3) / 2
        # So we need S >= 2 * max(n . p) / sqrt(3)
        
        proj = all_vertices @ normals.T
        max_proj = np.max(proj)
        
        current_side = 2.0 * max_proj / math.sqrt(3)
        
        if current_side < best_side:
            best_side = current_side
            best_outer_angle = phi_deg
    
    # Now do a simple gradient-based optimization using smooth surrogate
    # We'll use the SDF-based smooth penalty
    temperature = 0.05
    learning_rate = 0.01
    outer_angle_rad = math.radians(best_outer_angle)
    cos_a, sin_a = math.cos(outer_angle_rad), math.sin(outer_angle_rad)
    R_outer = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    normals_local = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)
    ])
    normals = normals_local @ R_outer.T
    
    best_side_opt = best_side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    # Simple coordinate descent with smooth objective
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Compute current smooth objective
        all_vertices = np.vstack([_hex_vertices(centers[i], angles[i]) for i in range(N)])
        
        # Compute distance to boundary for each vertex
        proj = all_vertices @ normals.T
        # The constraint is: n . p <= S * sqrt(3)/2
        # So violation = n . p - S * sqrt(3)/2
        # We want to minimize the smooth max of (n . p) over all vertices and normals
        # which is equivalent to minimizing S
        max_val = _smooth_max(proj.ravel(), temperature)
        
        # Gradient with respect to each center and angle
        # This is complex, so let's do a simpler perturbative approach
        improved = False
        
        # Try small perturbations to centers
        for i in range(N):
            for dx, dy in [(-0.01, 0), (0.01, 0), (0, -0.01), (0, 0.01)]:
                new_centers = centers.copy()
                new_centers[i] += np.array([dx, dy])
                all_v_new = np.vstack([_hex_vertices(new_centers[k], angles[k]) for k in range(N)])
                proj_new = all_v_new @ normals.T
                max_new = _smooth_max(proj_new.ravel(), temperature)
                
                if max_new < max_val - 1e-4:
                    centers = new_centers
                    max_val = max_new
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try small perturbations to angles
        for i in range(N):
            for da in [-0.5, 0.5]:
                new_angles = angles.copy()
                new_angles[i] += da
                all_v_new = np.vstack([_hex_vertices(centers[k], new_angles[k]) for k in range(N)])
                proj_new = all_v_new @ normals.T
                max_new = _smooth_max(proj_new.ravel(), temperature)
                
                if max_new < max_val - 1e-4:
                    angles = new_angles
                    max_val = max_new
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # No improvement, reduce temperature
            temperature *= 0.9
            if temperature < 0.001:
                break
    
    # Recompute final side length
    all_vertices_final = np.vstack([_hex_vertices(centers[i], angles[i]) for i in range(N)])
    proj_final = all_vertices_final @ normals.T
    final_side = 2.0 * np.max(proj_final) / math.sqrt(3)
    
    # Safety margin
    final_side *= 1.00001
    
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
    
    return inner, (0.0, 0.0), final_side, best_outer_angle
# EVOLVE-BLOCK-END
