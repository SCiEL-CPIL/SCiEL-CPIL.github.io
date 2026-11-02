import numpy as np
import time
import random

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Get 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def get_hexagon_normals(angle_deg):
    """Get the 6 outward normals for a regular hexagon with given angle.
    For a regular hexagon, the outward normals are perpendicular to the edges.
    If vertices are at angles theta_i = angle + i*pi/3, the edges are between theta_i and theta_{i+1}.
    The normal to the edge between v_i and v_{i+1} points outward.
    For a regular hexagon centered at origin, the outward normal for the edge between
    vertex i and i+1 is at angle theta_i + pi/6 + pi/2 = theta_i + 2*pi/3? No.
    
    Let's re-derive.
    Vertices at angles 0, 60, 120, 180, 240, 300 degrees (for angle=0).
    Edge 0: from v0 (1,0) to v1 (0.5, sqrt(3)/2). Midpoint: (0.75, sqrt(3)/4).
    Normal direction: (sqrt(3)/4, 0.75) normalized -> (sqrt(3)/2, 1/2) which is angle 30 degrees.
    So normal for edge i is at angle angle_deg + 30 + i*60 degrees.
    """
    base_angle = np.radians(angle_deg + 30)
    normals = []
    for i in range(6):
        theta = base_angle + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    return np.array(normals)

def sat_overlap(v1, n1, v2, n2):
    """Check if two hexagons overlap using Separating Axis Theorem.
    v1, v2: vertices (6x2 arrays)
    n1, n2: normals (6x2 arrays)
    Returns: True if overlapping, False if separated.
    Also returns the minimum distance if separated (for penalty).
    """
    min_overlap = 0.0
    min_dist = np.inf
    
    # Test axes from both hexagons
    all_normals = np.vstack([n1, n2])
    
    for normal in all_normals:
        # Project v1 onto normal
        proj1 = np.dot(v1, normal)
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        # Project v2 onto normal
        proj2 = np.dot(v2, normal)
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            # Separated on this axis
            if max1 < min2:
                dist = min2 - max1
            else:
                dist = min1 - max2
            min_dist = min(min_dist, dist)
            return False, min_dist
        
        # Overlap on this axis
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = max(min_overlap, overlap)
    
    return True, 0.0

def check_containment(inner_verts, outer_verts, outer_normals):
    """Check if all inner hexagons are contained within the outer hexagon.
    Returns: True if contained, False otherwise.
    Also returns the maximum violation distance.
    """
    max_violation = 0.0
    
    for iv in inner_verts:
        for normal in outer_normals:
            # Project outer hexagon onto normal
            proj_outer = np.dot(outer_verts, normal)
            min_outer = np.min(proj_outer)
            max_outer = np.max(proj_outer)
            
            # Project inner hexagon onto normal
            proj_inner = np.dot(iv, normal)
            min_inner = np.min(proj_inner)
            max_inner = np.max(proj_inner)
            
            # Check if inner is within outer bounds
            if min_inner < min_outer:
                violation = min_outer - min_inner
                max_violation = max(max_violation, violation)
            if max_inner > max_outer:
                violation = max_inner - max_outer
                max_violation = max(max_violation, violation)
    
    return max_violation == 0.0, max_violation

def evaluate_configuration(inner, outer_center, outer_side, outer_angle):
    """Evaluate a configuration: returns (side_length, penalty).
    penalty = containment_violation^2 + overlap_penalty
    """
    # Get outer hexagon vertices and normals
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    outer_normals = get_hexagon_normals(outer_angle)
    
    # Get inner hexagons
    inner_verts = []
    inner_normals = []
    for row in inner:
        x, y, ang = row[0], row[1], row[2]
        v = hexagon_vertices(x, y, 1.0, ang)
        n = get_hexagon_normals(ang)
        inner_verts.append(v)
        inner_normals.append(n)
    
    # Check containment
    contained, containment_violation = check_containment(inner_verts, outer_verts, outer_normals)
    
    # Check overlaps
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            overlapping, dist = sat_overlap(inner_verts[i], inner_normals[i], 
                                            inner_verts[j], inner_normals[j])
            if not overlapping:
                # If separated, no penalty
                pass
            else:
                # Overlapping, add penalty based on depth of overlap
                # We don't have the exact overlap depth from sat_overlap, 
                # so use a fixed penalty or recompute
                # For simplicity, use a large penalty
                overlap_penalty += 100.0
    
    total_penalty = containment_violation ** 2 + overlap_penalty
    return outer_side + total_penalty

def make_honeycomb_init(scale=1.0):
    """Create a honeycomb arrangement of 12 hexagons."""
    inner = []
    dx = np.sqrt(3) * scale
    dy = 1.5 * scale
    
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y, 0.0])
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    return positions

def estimate_outer_size(inner):
    """Estimate a good outer side length from inner positions."""
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    margin = 1.0
    max_x = max(abs(x) for x in xs) + margin
    max_y = max(abs(y) for y in ys) + margin
    side = max(max_x, max_y * 2 / np.sqrt(3)) + 0.5
    return side

def construct():
    start_time = time.time()
    
    # Start with honeycomb initialization
    inner = make_honeycomb_init(1.0)
    outer_center = [0.0, 0.0]
    outer_side = estimate_outer_size(inner)
    outer_angle = 0.0
    
    # Simulated Annealing
    current_config = [inner, outer_center, outer_side, outer_angle]
    current_score = evaluate_configuration(inner, outer_center, outer_side, outer_angle)
    
    best_config = [inner.copy(), outer_center.copy(), outer_side, outer_angle]
    best_score = current_score
    
    T = 1.0  # Initial temperature
    cooling = 0.995
    
    # Run SA until deadline
    while time.time() - start_time < DEADLINE:
        # Generate neighbor
        new_inner = [row[:] for row in inner]
        new_outer_center = outer_center[:]
        new_outer_side = outer_side
        new_outer_angle = outer_angle
        
        # Random move
        move_type = random.random()
        if move_type < 0.7:
            # Move an inner hexagon
            idx = random.randint(0, N-1)
            delta_x = random.gauss(0, 0.5)
            delta_y = random.gauss(0, 0.5)
            new_inner[idx][0] += delta_x
            new_inner[idx][1] += delta_y
        elif move_type < 0.85:
            # Rotate an inner hexagon
            idx = random.randint(0, N-1)
            delta_angle = random.gauss(0, 10)
            new_inner[idx][2] += delta_angle
        elif move_type < 0.95:
            # Change outer side
            new_outer_side += random.gauss(0, 0.3)
            new_outer_side = max(1.0, new_outer_side)
        else:
            # Change outer angle
            new_outer_angle += random.gauss(0, 5)
        
        # Evaluate
        new_score = evaluate_configuration(new_inner, new_outer_center, new_outer_side, new_outer_angle)
        
        # Accept or reject
        if new_score < current_score:
            current_config = [new_inner, new_outer_center, new_outer_side, new_outer_angle]
            current_score = new_score
            if new_score < best_score:
                best_config = [new_inner, new_outer_center, new_outer_side, new_outer_angle]
                best_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0:
                delta = 0
            if random.random() < np.exp(-delta / T):
                current_config = [new_inner, new_outer_center, new_outer_side, new_outer_angle]
                current_score = new_score
        
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Return best configuration
    inner_best, center_best, side_best, angle_best = best_config
    return inner_best, (center_best[0], center_best[1]), side_best, angle_best
