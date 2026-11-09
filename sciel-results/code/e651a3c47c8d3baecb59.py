import numpy as np
import time

def _hex_vertices(cx, cy, d, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    Assumes inner hexagons are aligned with angle 0 (standard orientation).
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All inner hexagon vertices
    # positions: (12, 2)
    # unit_hex_verts: (6, 2)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    
    # Outward normals of the outer hexagon edges
    # Edge i connects vertex i and i+1.
    # The normal to the edge between vertex i and i+1 points outward.
    # Vertex i is at angle a + 60i. Vertex i+1 is at angle a + 60(i+1).
    # Midpoint angle is a + 60i + 30.
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals) # (6, 2)
    
    # Project all vertices onto normals
    projections = all_verts_flat @ normals.T # (36, 6)
    max_projections = np.max(projections, axis=0) # (6,)
    
    # For a regular hexagon of side s, the distance from center to edge (apothem) is s * sqrt(3) / 2.
    # So max_proj <= s * sqrt(3) / 2
    # s >= 2 * max_proj / sqrt(3)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    Returns True if valid, False otherwise.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        # Check unit edge length
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False
        inner_verts_list.append(verts)

    # Check separation (SAT)
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            
            separated = False
            # Check axes from A
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            # Check axes from B
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    # Check inner vertices inside outer
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def _is_valid_configuration(positions):
    """
    Fast check if positions form a valid non-overlapping packing.
    Uses SAT for unit hexagons at angle 0.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    inner_verts_list = [positions[i] + unit_hex_verts for i in range(12)]
    
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            
            separated = False
            # Check axes from A
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            # Check axes from B
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Define lattice vectors for two orientations
    # Orientation 0: Flat top (standard)
    # Basis vectors for triangular lattice of side 1 (hexagons touching)
    # Distance between centers of touching unit hexagons is 2 * apothem = sqrt(3)
    # Wait, side length of hexagon is 1. Apothem is sqrt(3)/2.
    # Distance between centers of two touching unit hexagons is 2 * (sqrt(3)/2) = sqrt(3).
    # So lattice constant a = sqrt(3).
    
    # Triangular lattice basis for orientation 0 (aligned with x-axis)
    a = np.sqrt(3)
    v1_0 = np.array([a, 0])
    v2_0 = np.array([a/2, a*np.sqrt(3)/2]) # a * (1/2, sqrt(3)/2)
    
    # Orientation 1: Rotated by 30 degrees? Or just a different lattice orientation.
    # Actually, the hexagons themselves are fixed at angle 0.
    # The lattice can be rotated.
    # Let's try rotating the lattice basis by 0, 10, 20, 30 degrees.
    
    lattice_rotations = [0, 10, 20, 30, 40, 50]
    
    # Generate candidate integer coordinates
    # We need 12 points.
    # A hexagonal cluster of 12 points doesn't exist perfectly (1+6=7, 1+6+12=19).
    # So we take a subset of a larger lattice.
    # Let's generate a grid of points and pick the best 12.
    
    # To keep computation tractable, we will generate a set of candidate points
    # in a roughly circular region and select subsets.
    # However, choosing subsets is combinatorial.
    
    # Alternative: Use the known optimal or near-optimal configurations.
    # The seed program found a configuration with s ~ 1/1.1421 ~ 0.875? No, 1/s = 1.1421 => s ~ 0.875.
    # Wait, s must be >= sqrt(12) ~ 3.464.
    # 1/3.464 = 0.288.
    # Current score 1.1421 => s = 1/1.1421 = 0.875?
    # Let's check the seed code.
    # The seed code returns s. The score is 1/s.
    # If score is 1.1421, s is ~0.875.
    # But s must be >= sqrt(12) ~ 3.464.
    # There is a contradiction in my understanding of the score or the seed code.
    # Let's re-read the task: "The value is proportional to 1/s (higher is better)."
    # "Invalid: ... s < sqrt(12)".
    # So s >= 3.464.
    # Max possible 1/s is 1/3.464 = 0.288.
    # But the current program has score 1.1421.
    # This implies the "score" in the prompt is NOT 1/s directly, or I am misinterpreting "score".
    # "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # So if seed score is 1.0, and current is 1.1421, the current value is 1.1421 * seed_value.
    # If seed_value is 1/s_seed, then current_value = 1/s_current.
    # This implies s_current < s_seed.
    # But s >= 3.464.
    # So 1/s <= 0.288.
    # How can the score be 1.14?
    # Maybe the "score" reported in the prompt is the *relative* score compared to a baseline?
    # "search score is value/seed, so 1.0 equals the seed program".
    # So the *value* returned by the evaluator is 1/s.
    # If the seed program had s_seed, its value was 1/s_seed.
    # The current program has value 1.1421 * (1/s_seed).
    # This doesn't change the fact that 1/s is small.
    # Unless... the "score" 1.1421 is the *value* itself?
    # If value = 1.1421, then s = 0.875. But s >= 3.464.
    # This is impossible.
    
    # Let's look at the seed code again.
    # It returns `best_s`.
    # Maybe the task description "s < sqrt(12)" is a lower bound, but the *value* is defined differently?
    # "The value is proportional to 1/s".
    # If the proportionality constant is not 1?
    # Or maybe `s` in the return is not the side length?
    # "outer a (3,) array ... s its side length."
    # "Invalid: ... s < sqrt(12)".
    
    # Let's assume the "score" 1.1421 is the *relative* improvement over a baseline that might have a different definition, or the baseline s was large.
    # If baseline s = 3.464 (minimum), value = 0.288.
    # If current value = 1.1421, that's impossible for 1/s.
    
    # Wait, look at the inspirations: "score 1.3333".
    # If score is 1.3333, and it's 1/s, then s = 0.75.
    # This is definitely less than sqrt(12).
    
    # Is it possible that `s` in the return is NOT the side length of the outer hexagon, but something else?
    # "outer a (3,) array (x, y, angle_deg) for the outer hexagon, s its side length."
    
    # Let's re-read carefully: "Invalid: ... s < sqrt(12)".
    # This constraint MUST be satisfied.
    # So s >= 3.464.
    # So 1/s <= 0.288.
    
    # How can the score be 1.14?
    # Perhaps the "score" in the prompt is NOT the value 1/s, but the *ratio* to a specific seed?
    # "The search score is value/seed, so 1.0 equals the seed program and above 1.0 is an improvement".
    # So if the seed program had value V_seed, and the current program has value V_curr.
    # Score = V_curr / V_seed = 1.1421.
    # This means V_curr = 1.1421 * V_seed.
    # If V = 1/s, then 1/s_curr = 1.1421 * (1/s_seed) => s_curr = s_seed / 1.1421.
    # This implies s_curr < s_seed.
    # This is consistent.
    # So the *value* is 1/s. The *score* is relative.
    # The prompt says: "The evaluator measures the raw value... The search score is value/seed".
    # So the number 1.1421 is the *relative* score.
    # The *raw value* would be 1/s.
    # So if I want to beat 1.1421, I need to make 1/s larger, i.e., s smaller.
    
    # Okay, so my goal is to minimize s.
    # The minimum possible s is sqrt(12) ~ 3.464.
    # The seed program likely has s close to some value.
    # If the seed score is 1.0, and current is 1.1421, then s_current = s_seed / 1.1421.
    
    # Let's proceed with minimizing s.
    
    # Strategy:
    # 1. Generate candidate positions on a triangular lattice.
    # 2. For each candidate set of 12 positions, check if they are non-overlapping.
    # 3. If valid, compute min_outer_side for various angles.
    # 4. Keep the best s.
    
    # To generate candidate sets:
    # We can use a "greedy" or "local search" on the lattice.
    # Start with a known good configuration (e.g., the seed's initial guess).
    # Then try swapping points or moving points to nearby lattice sites.
    
    # Let's use a simple local search on the lattice.
    
    # Initial guess from seed:
    h = np.sqrt(3)/2
    init_pos = np.array([
        [0, 0],
        [2*h, 0],
        [h, 1.5],
        [-h, 1.5],
        [-2*h, 0],
        [-h, -1.5],
        [h, -1.5],
        [3*h, 1.5],
        [0, 3],
        [-3*h, 1.5],
        [-3*h, -1.5],
        [3*h, -1.5]
    ])
    
    # Convert to lattice coordinates?
    # The lattice vectors are v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5).
    # Note: 1.5 = sqrt(3) * sqrt(3)/2.
    # So v2 = (sqrt(3)/2, sqrt(3)*sqrt(3)/2).
    # Let's define the lattice basis properly.
    # a = sqrt(3).
    # v1 = (a, 0).
    # v2 = (a/2, a*sqrt(3)/2) = (sqrt(3)/2, 1.5).
    
    # We will represent positions as integer combinations of v1 and v2.
    # pos = i * v1 + j * v2.
    
    # Let's find the integer coordinates of the initial positions.
    # This might not be exact if the initial positions are not on the lattice.
    # But the initial positions ARE on the lattice.
    # (0,0) -> (0,0)
    # (2*h, 0) = (sqrt(3), 0) -> (1, 0)
    # (h, 1.5) = (sqrt(3)/2, 1.5) -> (0, 1)
    # (-h, 1.5) = (-sqrt(3)/2, 1.5) -> (-1, 1)
    # (-2*h, 0) = (-sqrt(3), 0) -> (-1, 0)
    # (-h, -1.5) = (-sqrt(3)/2, -1.5) -> (0, -1)
    # (h, -1.5) = (sqrt(3)/2, -1.5) -> (1, -1)
    # (3*h, 1.5) = (3*sqrt(3)/2, 1.5). 
    #   3*sqrt(3)/2 = 1.5 * sqrt(3).
    #   v1 = (sqrt(3), 0). v2 = (sqrt(3)/2, 1.5).
    #   i*sqrt(3) + j*sqrt(3)/2 = 3*sqrt(3)/2 => 2i + j = 3.
    #   j*1.5 = 1.5 => j = 1.
    #   2i + 1 = 3 => 2i = 2 => i = 1.
    #   So (1, 1).
    # (0, 3) = (0, 3).
    #   j*1.5 = 3 => j = 2.
    #   i*sqrt(3) + 2*sqrt(3)/2 = 0 => i*sqrt(3) + sqrt(3) = 0 => i = -1.
    #   So (-1, 2).
    # (-3*h, 1.5) = (-3*sqrt(3)/2, 1.5).
    #   j = 1.
    #   2i + 1 = -3 => 2i = -4 => i = -2.
    #   So (-2, 1).
    # (-3*h, -1.5) = (-3*sqrt(3)/2, -1.5).
    #   j = -1.
    #   2i - 1 = -3 => 2i = -2 => i = -1.
    #   So (-1, -1).
    # (3*h, -1.5) = (3*sqrt(3)/2, -1.5).
    #   j = -1.
    #   2i - 1 = 3 => 2i = 4 => i = 2.
    #   So (2, -1).
    
    init_coords = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [0, -1],
        [1, -1],
        [1, 1],
        [-1, 2],
        [-2, 1],
        [-1, -1],
        [2, -1]
    ])
    
    # Local search on lattice coordinates
    # Moves: change one coordinate of one point by +-1 in i or j.
    
    best_coords = init_coords.copy()
    best_s = _min_outer_side(best_coords @ np.array([[a, a/2], [0, 1.5]].T).T, 0) # Wait, need to convert coords to pos
    # pos = i * v1 + j * v2
    # v1 = (a, 0), v2 = (a/2, 1.5)
    # pos_x = i*a + j*a/2
    # pos_y = j*1.5
    
    def coords_to_pos(coords):
        i = coords[:, 0]
        j = coords[:, 1]
        x = i * a + j * a / 2
        y = j * 1.5
        return np.column_stack([x, y])
    
    best_pos = coords_to_pos(best_coords)
    best_s = _min_outer_side(best_pos, 0)
    
    # Optimize angle for this configuration
    best_angle = 0
    for ang in np.linspace(0, 60, 61):
        s_test = _min_outer_side(best_pos, ang)
        if s_test < best_s:
            best_s = s_test
            best_angle = ang
            
    best_score = 1.0 / best_s
    
    # Simple local search
    num_iterations = 5000
    for _ in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Pick a random point
        idx = np.random.randint(12)
        # Pick a random move
        di = np.random.choice([-1, 0, 1])
        dj = np.random.choice([-1, 0, 1])
        if di == 0 and dj == 0:
            continue
            
        new_coords = best_coords.copy()
        new_coords[idx] += [di, dj]
        
        # Check if this coordinate is already occupied
        # If yes, skip (or handle swap, but simple rejection is fine for now)
        if np.any(np.all(new_coords == new_coords[idx], axis=1)[:-1]) or np.any(np.all(new_coords[1:] == new_coords[idx], axis=1)):
             # More efficient check:
             pass
        
        # Actually, let's just check validity of the new configuration
        new_pos = coords_to_pos(new_coords)
        
        # Fast overlap check
        if not _is_valid_configuration(new_pos):
            continue
            
        # Calculate new s
        # Try a few angles
        min_s_new = np.inf
        for ang in [best_angle, best_angle + 5, best_angle - 5, 0, 15, 30, 45]:
            s_test = _min_outer_side(new_pos, ang)
            if s_test < min_s_new:
                min_s_new = s_test
                
        if min_s_new < best_s:
            best_s = min_s_new
            best_coords = new_coords
            best_pos = new_pos
            best_score = 1.0 / best_s
    
    # Final polish: Optimize angle more precisely for the best configuration
    if time.time() < deadline - 10:
        # Bisection or fine grid for angle
        best_angle_final = 0
        best_s_final = np.inf
        for ang in np.linspace(0, 60, 601):
            s_test = _min_outer_side(best_pos, ang)
            if s_test < best_s_final:
                best_s_final = s_test
                best_angle_final = ang
        best_s = best_s_final
        best_score = 1.0 / best_s
        
    # Construct return values
    inner = np.zeros((12, 3))
    inner[:, :2] = best_pos
    inner[:, 2] = 0.0 # Inner hexagons aligned with 0
    
    outer = np.array([0, 0, best_angle_final if 'best_angle_final' in locals() else best_angle])
    
    # Check validity
    if _check_validity(inner, outer, best_s):
        return inner, outer, best_s
    else:
        # Fallback to seed's initial guess
        inner_fallback = np.zeros((12, 3))
        inner_fallback[:, :2] = init_pos
        inner_fallback[:, 2] = 0.0
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(init_pos, 0)
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
