# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _separation_violation(verts1, verts2):
    """
    Using SAT: if there exists a separating axis (edge normal of either hexagon)
    where projection intervals are disjoint (gap > 0), then they are separated.
    We return the maximum violation (how much they overlap along the best axis).
    If all axes show separation (gap > 0), violation is negative.
    We want violation <= 0 for valid separation.
    """
    # Edge normals for hex1
    n1 = []
    for i in range(6):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
        n1.append(normal)
    
    n2 = []
    for i in range(6):
        p1 = verts2[i]
        p2 = verts2[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
        n2.append(normal)
    
    all_normals = n1 + n2
    
    # For each normal, compute projection intervals
    max_separation = -np.inf  # We want the axis with maximum gap (most separated)
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Gap: if intervals are disjoint, gap is positive
        # gap = min(max1, max2) - max(min1, min2) but we need the distance between intervals
        # If max1 < min2: gap = min2 - max1 (separated)
        # If max2 < min1: gap = min1 - max2 (separated)
        # Else: overlap, gap is negative
        gap = min(max1, max2) - max(min1, min2)
        
        if gap > max_separation:
            max_separation = gap
    
    # max_separation is the maximum gap over all axes.
    # If max_separation > 0, they are separated by that amount.
    # If max_separation <= 0, they overlap.
    # The violation is -max_separation. If violation < 0, they are separated.
    return -max_separation

def _inside_outer_violation(vert, outer_verts):
    """
    Check if vert is outside the outer hexagon (counter-clockwise vertices).
    For each edge (p1, p2), cross(p2-p1, vert-p1) should be >= 0 for CCW.
    Return the minimum cross product (most negative is worst violation).
    """
    min_cross = np.inf
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        v = vert - p1
        cross = edge[0] * v[1] - edge[1] * v[0]
        if cross < min_cross:
            min_cross = cross
    return min_cross

def _compute_objective_and_violations(positions, outer_s, outer_angle):
    """
    positions: (12, 2) array of centers. All inner hexagons have angle 0 and side 1.
    outer_s: side length of outer hexagon.
    outer_angle: angle of outer hexagon.
    Returns: objective (max distance of inner vertices from origin), and a list of violations.
    """
    # Inner hexagons have side 1, angle 0
    inner_verts_list = []
    for i in range(12):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        inner_verts_list.append(verts)
    
    # Outer hexagon
    outer_verts = _hex_vertices(0.0, 0.0, outer_s, outer_angle)
    
    # Objective: max distance from origin of any inner vertex
    max_dist = 0.0
    for verts in inner_verts_list:
        dists = np.linalg.norm(verts, axis=1)
        max_dist = max(max_dist, np.max(dists))
    
    # Violations:
    # 1. Separation between inner hexagons
    sep_violations = []
    for i in range(12):
        for j in range(i + 1, 12):
            viol = _separation_violation(inner_verts_list[i], inner_verts_list[j])
            sep_violations.append(viol)
    
    # 2. Inside outer hexagon
    inside_violations = []
    for verts in inner_verts_list:
        for v in verts:
            viol = _inside_outer_violation(v, outer_verts)
            inside_violations.append(viol)
    
    return max_dist, sep_violations, inside_violations

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds max
    
    # Initial configuration: a hexagonal grid of 12 hexagons
    # Center at (0,0), 6 around it, and 5 more in the second ring (to make 12)
    # Actually, a standard hexagonal packing of 12:
    # Ring 0: 1 hex (center)
    # Ring 1: 6 hexes
    # Ring 2: 5 hexes (to total 12)
    
    # Let's use a simple grid first and then optimize.
    # Positions for 12 hexagons in a hexagonal grid:
    # (0,0)
    # (1,0), (-1,0), (0.5, 0.866), (-0.5, 0.866), (0.5, -0.866), (-0.5, -0.866)
    # (2,0), (-2,0), (1.5, 0.866), (-1.5, 0.866), (1.5, -0.866), (-1.5, -0.866) -> too many
    # Let's just pick 12 positions that form a compact cluster.
    
    # A better initial guess: a 3x4 grid or similar, but hexagonal is better.
    # Let's use the positions from the seed program but scaled down.
    # Seed positions:
    # [0, 0], [-2.5, 0], [2.5, 0], [-1.25, 2.17], [1.25, 2.17], [-1.25, -2.17], [1.25, -2.17],
    # [-3.75, 2.17], [3.75, 2.17], [-3.75, -2.17], [3.75, -2.17], [0, -4]
    
    # Let's try a tighter initial packing.
    # A hexagonal number 12 is not a perfect hex number (1, 7, 19).
    # So we need a non-symmetric shape.
    
    # Let's use a simple optimization approach:
    # 1. Start with a reasonable guess.
    # 2. Use Nelder-Mead to minimize the outer hexagon side length.
    
    # We'll parameterize the positions of the 12 hexagons (24 variables) and the outer hexagon side length (1 variable).
    # Total 25 variables.
    
    # Initial guess:
    # Let's place them in a 2x6 grid or similar.
    # Actually, let's use the seed positions as a starting point, but we'll optimize.
    
    initial_positions = np.array([
        [0.0, 0.0],
        [-2.5, 0.0],
        [2.5, 0.0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17],
        [0.0, -4.0]
    ])
    
    # We'll optimize the positions and the outer side length.
    # Objective: minimize outer_s, subject to constraints.
    # We'll use a penalty method or just minimize outer_s + penalty * max_violation.
    
    def objective(params):
        positions = params[:24].reshape(12, 2)
        outer_s = params[24]
        
        if outer_s < math.sqrt(12):
            return 1e10  # Penalty for too small outer hexagon
        
        max_dist, sep_violations, inside_violations = _compute_objective_and_violations(positions, outer_s, 0.0)
        
        # The outer hexagon must contain all inner vertices.
        # The max distance from origin of any inner vertex must be <= the inradius of the outer hexagon.
        # Inradius of outer hexagon = outer_s * sqrt(3) / 2
        # But the outer hexagon is rotated, so we need to check the actual containment.
        # We already compute inside_violations, which checks containment.
        
        max_sep_viol = max(sep_violations) if sep_violations else 0.0
        max_inside_viol = max(inside_violations) if inside_violations else 0.0
        
        # We want max_sep_viol <= 0 and max_inside_viol >= 0.
        # Penalty for violations:
        penalty = 0.0
        if max_sep_viol > 0:
            penalty += max_sep_viol * 10.0
        if max_inside_viol < 0:
            penalty += (-max_inside_viol) * 10.0
        
        return outer_s + penalty
    
    # Initial parameters:
    initial_params = np.concatenate([initial_positions.flatten(), [8.0]])
    
    # Use Nelder-Mead to minimize the objective.
    from scipy.optimize import minimize
    
    best_result = None
    best_score = 0.0
    
    # We'll run multiple optimizations with different initial guesses.
    # But first, let's try the initial guess.
    
    try:
        result = minimize(objective, initial_params, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.success:
            positions = result.x[:24].reshape(12, 2)
            outer_s = result.x[24]
            
            # Check validity
            max_dist, sep_violations, inside_violations = _compute_objective_and_violations(positions, outer_s, 0.0)
            max_sep_viol = max(sep_violations) if sep_violations else 0.0
            max_inside_viol = max(inside_violations) if inside_violations else 0.0
            
            if max_sep_viol <= 1e-6 and max_inside_viol >= -1e-6 and outer_s >= math.sqrt(12):
                score = 1.0 / outer_s
                if score > best_score:
                    best_score = score
                    best_result = (positions, outer_s)
    except Exception as e:
        pass
    
    # If we have a valid result, return it.
    if best_result is not None:
        positions, outer_s = best_result
        inner_hex_data = np.column_stack([positions, np.zeros(12)])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, outer_s
    
    # If no valid result, return the initial configuration with a large outer hexagon.
    inner_hex_data = np.column_stack([initial_positions, np.zeros(12)])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_s = 8.0
    return inner_hex_data, outer_hex_data, outer_s

# EVOLVE-BLOCK-END
