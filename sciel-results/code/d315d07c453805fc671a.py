"""
Packing 12 unit hexagons in a minimal regular hexagon.
Uses Simulated Annealing with a force-based energy model.
"""
import numpy as np
import time
import math

N = 12
DEADLINE = 145.0  # Safety margin before 150s kill

# Pre-compute unit hexagon vertices for standard orientation (flat-top? pointy-top?)
# Standard math definition: vertex at angle 0 is (1, 0).
# Let's use pointy-top for standard "hexagon" visual, but orientation is a variable.
# Actually, the problem says "regular hexagon". Orientation is a DOF.
# We will define vertices relative to center (0,0) with side 1.
def get_unit_hex_verts(angle_deg):
    """
    Returns vertices of a unit hexagon (side=1) rotated by angle_deg.
    Vertices are at distance 1 from center? No, side length is 1.
    Distance from center to vertex (circumradius) = side = 1.
    Distance from center to edge (apothem) = sqrt(3)/2.
    """
    a = np.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 relative to rotation
    thetas = a + np.radians(np.arange(0, 360, 60))
    x = np.cos(thetas)
    y = np.sin(thetas)
    return np.column_stack((x, y))

def get_hex_verts(center, angle_deg):
    """
    Returns vertices of a unit hexagon at given center and angle.
    """
    cx, cy = center
    verts = get_unit_hex_verts(angle_deg)
    return verts + np.array([cx, cy])

def sat_min_separation(c1, c2):
    """
    Computes minimum separation between two hexagons.
    c1, c2: [x, y, angle_deg]
    Returns: (min_sep, axis)
    min_sep > 0: separated by min_sep
    min_sep < 0: overlapping by -min_sep
    """
    v1 = get_hex_verts((c1[0], c1[1]), c1[2])
    v2 = get_hex_verts((c2[0], c2[1]), c2[2])
    
    # SAT Axes: Normals to the edges of both hexagons
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-10:
                continue
            # Normal vector
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
            
    min_sep = float('inf')
    best_axis = None
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Separation on this axis
        # If max1 < min2, separation is min2 - max1
        # If max2 < min1, separation is min1 - max2
        # If they overlap on this axis, separation is negative
        sep = max(min2 - max1, min1 - max2)
        
        if sep < min_sep:
            min_sep = sep
            best_axis = axis
            
    return min_sep, best_axis

def required_outer_side(centers, outer_center, outer_angle):
    """
    Computes the minimum side length of a regular hexagon centered at outer_center
    and rotated by outer_angle to contain all inner hexagons.
    """
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    # Rotation matrix to align outer hexagon with world axes for checking
    # We want to project points onto the 6 normals of the outer hexagon.
    # The normals of a regular hexagon with angle 0 are at 0, 60, 120, ...
    # If the outer hexagon is rotated by outer_angle, its normals are rotated too.
    # Instead of rotating points, we can rotate the normals.
    # Normal k is at angle k*60 + outer_angle.
    
    # Optimization: Pre-compute normal vectors for the 6 directions
    # But outer_angle changes in optimization? 
    # In this SA, we fix outer_angle to 0 for simplicity? 
    # No, the task allows rotation of the outer hexagon.
    # However, due to symmetry, we can fix outer_angle to 0 without loss of generality
    # IF we allow inner hexagons to rotate freely? 
    # Actually, fixing outer angle to 0 is valid because any solution with rotated outer
    # can be rotated back to 0 by rotating all inner hexagons and centers accordingly.
    # So we can fix outer_angle = 0.0 to reduce DOF and speed up search.
    
    # Let's verify: If we have a valid packing in a hexagon of angle A,
    # we can rotate the entire system by -A. The outer hexagon becomes angle 0.
    # The inner hexagons rotate by -A. Their relative positions and orientations
    # change, but the packing remains valid and the outer size is the same.
    # So yes, fix outer_angle = 0.
    
    # Normals for outer hexagon (angle 0):
    # Flat-top or Pointy-top?
    # Standard vertex at 0 deg: Pointy-top.
    # Edges are between (1,0) and (0.5, sqrt(3)/2).
    # Normal to edge between v0 and v1:
    # v0=(1,0), v1=(0.5, 0.866). Edge = (-0.5, 0.866). Normal = (-0.866, -0.5) or (0.866, 0.5).
    # The 6 outward normals for a pointy-top hexagon (vertex at 0) are at 30, 90, 150, 210, 270, 330?
    # Let's check.
    # Center (0,0). Vertex (1,0).
    # Apothem is distance to edge.
    # Edge connecting (1,0) and (0.5, sqrt(3)/2). Midpoint (0.75, sqrt(3)/4).
    # Vector from center to midpoint: (0.75, 0.433). Angle = arctan(0.433/0.75) = 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    
    # For a hexagon with side S, the apothem is S * sqrt(3) / 2.
    # A point P is inside if dot(P, normal_k) <= apothem for all k.
    
    normals = []
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    for (px, py, pa) in centers:
        verts = get_hex_verts((px, py), pa)
        # Relative to outer center
        rel_verts = verts - np.array(outer_center)
        
        for v in rel_verts:
            for n in normals:
                d = np.dot(v, n)
                if d > max_extent:
                    max_extent = d
                    
    # max_extent is the maximum projection onto the normals.
    # This equals the required apothem.
    # Side S = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def compute_energy_and_forces(centers, outer_center):
    """
    Computes energy (required_side + penalty) and forces for each inner hexagon.
    Returns: (energy, forces)
    forces: list of [fx, fy, fa] for each center.
    """
    n = len(centers)
    forces = [[0.0, 0.0, 0.0] for _ in range(n)]
    
    # 1. Containment Energy and Forces
    # We want to minimize required_side.
    # required_side is determined by the "furthest" vertex projection.
    # To get a smooth force, we can sum penalties for any vertex exceeding a "soft" boundary.
    # However, required_side itself is a non-smooth function (max of projections).
    # Let's use a penalty based on the current required_side.
    # Actually, simpler: Just compute required_side as the main energy component.
    # For forces, we can approximate the gradient of the max projection.
    # If vertex v is the one determining the max, it pushes outward.
    
    req_side = required_outer_side(centers, outer_center, 0.0)
    energy = req_side
    
    # To get forces for containment, we can calculate the "pressure" from the outer walls.
    # For each inner hexagon, check its vertices.
    # If a vertex is close to the boundary (projection > req_side * sqrt(3)/2 * 0.99),
    # apply an inward force.
    
    apothem_limit = req_side * np.sqrt(3) / 2
    normals = []
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
        
    for i, (px, py, pa) in enumerate(centers):
        verts = get_hex_verts((px, py), pa)
        rel_verts = verts - np.array(outer_center)
        
        fx = 0.0
        fy = 0.0
        
        for v in rel_verts:
            for n in normals:
                d = np.dot(v, n)
                if d > apothem_limit * 0.95: # Soft boundary
                    # Excess
                    excess = d - apothem_limit * 0.95
                    # Force is in direction -n
                    fx -= n[0] * excess
                    fy -= n[1] * excess
                    
        forces[i][0] += fx
        forces[i][1] += fy
        
    # 2. Overlap Energy and Forces
    # Penalty for overlaps
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep, axis = sat_min_separation(centers[i], centers[j])
            if sep < 0:
                # Overlapping
                overlap_amount = -sep
                overlap_penalty += overlap_amount ** 2
                
                # Force: push apart along axis
                # i moves in -axis direction, j in +axis direction
                push = overlap_amount * 2.0 # Strength factor
                if axis is not None:
                    forces[i][0] -= axis[0] * push
                    forces[i][1] -= axis[1] * push
                    forces[j][0] += axis[0] * push
                    forces[j][1] += axis[1] * push
                    
    energy += 100.0 * overlap_penalty
    
    return energy, forces

def check_validity(centers, outer_side):
    """
    Strict check for validity.
    """
    # Check containment
    req_side = required_outer_side(centers, (0,0), 0.0)
    if req_side > outer_side + 1e-7:
        return False
        
    # Check overlaps
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            sep, _ = sat_min_separation(centers[i], centers[j])
            if sep < -1e-7:
                return False
    return True

def initial_config_hexagonal():
    """
    Hexagonal lattice arrangement.
    Spacing: For unit hexagons (side 1), the distance between centers of touching hexagons
    in a flat lattice is 2 * apothem = sqrt(3) if edge-to-edge?
    No.
    If two hexagons touch edge-to-edge, the distance between centers is 2 * apothem = sqrt(3).
    If they touch vertex-to-vertex?
    Standard hexagonal tiling:
    Center to center distance = sqrt(3) * side?
    Let's check.
    Apothem = sqrt(3)/2.
    Distance between parallel edges = sqrt(3).
    So if we stack them, the center-to-center distance in the "tight" direction is sqrt(3).
    In the other direction (along the edge), it's also sqrt(3)?
    No, the lattice vectors for a hexagonal tiling of unit hexagons are:
    a1 = (sqrt(3), 0) ? No.
    Let's use the vertices.
    Center (0,0). Vertices at (1,0), (0.5, 0.866)...
    Neighbor to the right (touching edge x=0.5 to x=1? No, edge is vertical? No.)
    For pointy-top hexagon:
    Rightmost point is (1,0). Leftmost is (-1,0).
    Topmost is (0, 0.866)? No, (0.5, 0.866) and (-0.5, 0.866) are top vertices.
    Wait, (1,0) is a vertex. (0.5, 0.866) is a vertex.
    Edge between them.
    The neighbor to the "North-East" shares the edge between (1,0) and (0.5, 0.866).
    The center of that neighbor is at distance 2 * apothem = sqrt(3) from origin,
    in the direction of the normal to that edge.
    Normal to edge (1,0)-(0.5, 0.866) is at 30 deg.
    So neighbor center is at sqrt(3) * (cos(30), sin(30)) = sqrt(3) * (0.866, 0.5) = (1.5, 0.866).
    
    So the lattice vectors are:
    v1 = (1.5, 0.866)
    v2 = (1.5, -0.866)
    v3 = (0, 1.732) = v1 - v2? No.
    v1 + v2 = (3, 0).
    v1 - v2 = (0, 1.732).
    
    So we can generate positions using v1 and v2.
    """
    centers = []
    v1 = np.array([1.5, np.sqrt(3)/2])
    v2 = np.array([1.5, -np.sqrt(3)/2])
    
    # Generate a grid of indices
    # We want 12 points.
    # A cluster of 19 points (center + 6 + 12) is too many.
    # A cluster of 7 (center + 6) is too few.
    # Let's take a 3x4 rectangle or similar.
    # Or just take the 12 closest points to origin from the lattice.
    
    points = set()
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            points.add((round(p[0], 5), round(p[1], 5)))
            
    # Sort by distance to origin
    pts_list = sorted(points, key=lambda p: p[0]**2 + p[1]**2)
    selected = pts_list[:12]
    
    for p in selected:
        centers.append([p[0], p[1], 0.0])
        
    return centers

def initial_config_circular():
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def simulated_annealing(init_centers, time_limit):
    """
    Run SA on the given initial configuration.
    """
    start_time = time.time()
    centers = [list(c) for c in init_centers]
    outer_center = [0.0, 0.0] # Fixed
    
    # Initial energy
    energy, forces = compute_energy_and_forces(centers, outer_center)
    
    best_energy = float('inf')
    best_centers = None
    best_valid_side = float('inf')
    best_valid_centers = None
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9999
    
    # Step size
    step_size = 0.5
    
    while time.time() - start_time < time_limit:
        # Perturb
        new_centers = [list(c) for c in centers]
        for i in range(N):
            # Move
            new_centers[i][0] += np.random.uniform(-step_size, step_size)
            new_centers[i][1] += np.random.uniform(-step_size, step_size)
            # Rotate
            new_centers[i][2] += np.random.uniform(-10, 10)
            
        new_energy, _ = compute_energy_and_forces(new_centers, outer_center)
        
        # Acceptance
        delta = new_energy - energy
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            centers = new_centers
            energy = new_energy
            
            # Check validity for best tracking
            # This is expensive, so do it every few steps?
            # Or just track best energy and check validity at the end?
            # Better: Check validity periodically or if energy improves significantly.
            # Since we have a time limit, let's check validity if new_energy < best_energy.
            if new_energy < best_energy:
                best_energy = new_energy
                # Check if it's strictly valid
                # We need the required side for validity check
                req_side = required_outer_side(centers, outer_center, 0.0)
                if check_validity(centers, req_side):
                    if req_side < best_valid_side:
                        best_valid_side = req_side
                        best_valid_centers = [list(c) for c in centers]
                        
        # Cool
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Reheat or stop?
            # If we cool too much, we might get stuck.
            # Let's restart with a new random config if we have time?
            # For now, just continue with small T.
            
    return best_valid_side, best_valid_centers

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_centers = None
    
    # List of initial configurations
    configs = [
        initial_config_hexagonal(),
        initial_config_circular(),
        initial_config_random(1),
        initial_config_random(2),
        initial_config_random(3)
    ]
    
    # Allocate time
    # 5 configs.
    # Give each a chunk of time.
    # If one finishes early, use remaining time for the next?
    # Simpler: Run each for a fixed fraction, or until deadline.
    
    for i, config in enumerate(configs):
        # Check if we have enough time for a run
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5.0:
            break
            
        # Allocate time for this run
        # Let's give each run 25 seconds max.
        run_time_limit = min(25.0, remaining - 2.0)
        
        side, centers = simulated_annealing(config, run_time_limit)
        
        if centers is not None:
            if side < best_side:
                best_side = side
                best_centers = centers
                
    # Fallback if no valid solution found
    if best_centers is None:
        # Use a known valid but larger packing
        # e.g. 3 rows of 4, or just spread out
        centers = initial_config_hexagonal()
        # Expand them to ensure validity
        # Just compute required side
        best_side = required_outer_side(centers, (0,0), 0.0)
        # Ensure no overlaps by scaling positions?
        # The hexagonal lattice initial config should be valid if spacing is correct.
        # My lattice spacing was based on touching edges.
        # v1 = (1.5, 0.866). Distance = sqrt(1.5^2 + 0.866^2) = sqrt(2.25 + 0.75) = sqrt(3).
        # This is the touching distance. So they are disjoint.
        # So this should be valid.
        best_centers = centers
        # Re-verify
        if not check_validity(best_centers, best_side):
             # Force validity by spreading
             for c in best_centers:
                 c[0] *= 1.1
                 c[1] *= 1.1
             best_side = required_outer_side(best_centers, (0,0), 0.0)
             
    # Format output
    inner = []
    for c in best_centers:
        inner.append((c[0], c[1], c[2]))
        
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
