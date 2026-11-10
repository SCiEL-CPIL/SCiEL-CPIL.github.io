# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, 
    using numerical optimization to minimize the outer hexagon side length.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin for 600s limit
    
    # Constants
    n_inner = 11
    unit_side = 1.0
    
    # Helper to get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(np.arange(6) * 60)
        return np.column_stack((cx + side * np.cos(angles), cy + side * np.sin(angles)))
    
    # Helper to check if two hexagons intersect
    # Returns True if they intersect (overlap), False if separated
    def check_intersection(v1, v2):
        # v1, v2 are (6,2) arrays of vertices
        # Separation axis theorem for convex polygons
        # We check all 12 edge normals (6 from each)
        
        def get_edges_normals(vertices):
            edges = np.roll(vertices, -1, axis=0) - vertices
            # Normal is perpendicular to edge. For CCW polygon, normal is (dy, -dx) or (-dy, dx)
            # We just need the direction, sign doesn't matter for separation check
            normals = np.column_stack((edges[:, 1], -edges[:, 0]))
            # Normalize to unit length for consistent projection
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return normals / norms

        n1 = get_edges_normals(v1)
        n2 = get_edges_normals(v2)
        
        all_normals = np.vstack([n1, n2])
        
        for n in all_normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check if intervals are disjoint
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                # Separated on this axis
                return False
        
        # If no separating axis found, they intersect
        return True

    # Helper to calculate minimum side length of outer hexagon containing all inner hexes
    def calc_outer_side(inner_positions, inner_angles, outer_angle_deg):
        # inner_positions: (11, 2)
        # inner_angles: (11,)
        # outer_angle_deg: scalar
        
        # Get vertices of all inner hexes
        all_vertices = []
        for i in range(n_inner):
            v = get_hex_vertices(inner_positions[i, 0], inner_positions[i, 1], unit_side, inner_angles[i])
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        
        # Outer hexagon is centered at origin for calculation purposes, 
        # but we need to find the minimal side length S such that 
        # all vertices are inside a hexagon of side S rotated by outer_angle_deg.
        # 
        # A regular hexagon centered at origin with side S and angle A contains point (x,y)
        # if for all 6 face normals n_k, dot((x,y), n_k) <= S * sqrt(3)/2 (apothem).
        # Wait, apothem of hexagon with side S is S * sqrt(3)/2.
        # The distance from center to face is the apothem.
        
        a = np.deg2rad(outer_angle_deg)
        # Face normals are at angles a + 30 + 60k ? 
        # Vertices are at a + 60k. Faces are perpendicular to radius at a+30+60k.
        # So normals are at angles a + 30 + 60k.
        
        normal_angles = a + np.deg2rad(30 + np.arange(6) * 60)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Max projection of any vertex onto any normal
        # proj = v @ n
        # We need max over all v and n of dot(v, n)
        
        # Efficient way: for each normal, find max projection
        # (N, 2) @ (2, 6) -> (N, 6)
        projections = np.dot(all_vertices, normals)
        max_projections = np.max(projections, axis=0)
        
        # The required apothem is max(max_projections)
        required_apothem = np.max(max_projections)
        
        # Side S = apothem / (sqrt(3)/2) = 2 * apothem / sqrt(3)
        return 2.0 * required_apothem / np.sqrt(3.0)

    # Objective function for optimization
    def objective(params):
        # params: 11*3 + 1 = 34 variables
        # 0-32: inner hexes (x, y, angle)
        # 33: outer angle
        
        if time.time() > deadline - 5.0:
            return 1e10  # Timeout guard
            
        inner_params = params[:33].reshape(11, 3)
        outer_angle = params[33]
        
        inner_positions = inner_params[:, :2]
        inner_angles = inner_params[:, 2]
        
        # 1. Check intersections
        penalty = 0.0
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                v1 = get_hex_vertices(inner_positions[i, 0], inner_positions[i, 1], unit_side, inner_angles[i])
                v2 = get_hex_vertices(inner_positions[j, 0], inner_positions[j, 1], unit_side, inner_angles[j])
                if check_intersection(v1, v2):
                    # Estimate overlap depth for penalty? 
                    # Simple penalty: count intersections
                    penalty += 1.0
                    # Better: compute minimum separation violation
                    # For now, just a strong penalty
                    penalty += 10.0
        
        if penalty > 0:
            return penalty * 1000.0 + calc_outer_side(inner_positions, inner_angles, outer_angle)
            
        # 2. Calculate outer side
        side = calc_outer_side(inner_positions, inner_angles, outer_angle)
        
        # Add small penalty if side is too small (shouldn't happen if valid)
        return side

    # Initial guess: Honeycomb pattern
    # Center (0,0)
    # 6 around it
    # 4 further out? 
    # Let's use a grid-like start
    
    init_params = np.zeros(34)
    
    # Center
    init_params[0:3] = [0, 0, 0]
    
    # Ring 1 (distance ~ sqrt(3) for touching? No, touching distance is sqrt(3) for centers if aligned?)
    # Distance between centers of touching hexes (side 1) is sqrt(3).
    d1 = np.sqrt(3)
    angles_ring1 = np.deg2rad(np.arange(6) * 60)
    for i in range(6):
        idx = (i + 1) * 3
        init_params[idx] = d1 * np.cos(angles_ring1[i])
        init_params[idx+1] = d1 * np.sin(angles_ring1[i])
        init_params[idx+2] = 0 # angle
        
    # Ring 2? We need 11. 1 + 6 = 7. Need 4 more.
    # Place them in the "valleys" of the outer ring? Or extend the arms?
    # Let's just place them in a 2x2 block offset?
    # Or simply place them at distance 2*sqrt(3) in some directions?
    
    # Let's try a "flower" with 4 petals extended
    # Positions 7-10
    # Let's put them at angles 30, 150, 210, 330 at distance 2*sqrt(3)?
    # That might overlap with ring 1.
    # Distance from center to ring 1 is sqrt(3).
    # Distance from ring 1 to next shell is sqrt(3).
    # So distance 2*sqrt(3) is safe from center, but might clash with ring 1.
    # Angle 30 is between 0 and 60.
    # Point (2*sqrt(3) * cos(30), 2*sqrt(3) * sin(30)) = (3, sqrt(3))
    # Ring 1 point at 0 deg: (sqrt(3), 0).
    # Dist^2 = (3-1.73)^2 + (1.73)^2 = 1.62 + 3 = 4.62. sqrt(4.62) > sqrt(3). OK.
    
    d2 = 2 * np.sqrt(3)
    angles_ring2 = np.deg2rad([30, 150, 210, 330])
    for i in range(4):
        idx = (i + 7) * 3
        init_params[idx] = d2 * np.cos(angles_ring2[i])
        init_params[idx+1] = d2 * np.sin(angles_ring2[i])
        init_params[idx+2] = 0
        
    init_params[33] = 0 # Outer angle
    
    # Run optimization
    best_params = init_params
    best_score = objective(init_params)
    
    # Use Nelder-Mead or L-BFGS-B?
    # Nelder-Mead is robust for non-smooth (penalty) functions.
    
    try:
        # Time check
        if time.time() > deadline - 10.0:
            pass
        else:
            res = so.minimize(objective, init_params, method='Nelder-Mead', 
                              options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
            # Try another start? Maybe random perturbations
            # If time permits
            
    except Exception as e:
        pass
        
    # Extract best result
    inner_params = best_params[:33].reshape(11, 3)
    outer_angle = best_params[33]
    
    inner_hex_data = inner_params
    outer_hex_data = np.array([0, 0, outer_angle])
    outer_hex_side_length = calc_outer_side(inner_hex_data[:, :2], inner_hex_data[:, 2], outer_angle)
    
    # Ensure side is at least sqrt(11) as per contract? 
    # The contract says "side >= sqrt(11)". 
    # sqrt(11) ~ 3.31. 
    # Our packing is likely larger. 
    # If the optimizer finds a tiny side due to overlap penalty failure, we might return invalid.
    # We should verify validity before returning.
    
    # Quick validity check
    def is_valid(inner, outer, side):
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
            return False
        if side < np.sqrt(11):
            return False
            
        # Check intersections
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                v1 = get_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
                v2 = get_hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
                if check_intersection(v1, v2):
                    return False
        
        # Check containment
        # Outer hex centered at (0,0) with angle outer[2] and side 'side'
        a = np.deg2rad(outer[2])
        normal_angles = a + np.deg2rad(30 + np.arange(6) * 60)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        apothem = side * np.sqrt(3) / 2.0
        
        for i in range(n_inner):
            v = get_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            projections = np.dot(v, normals)
            if np.any(projections > apothem + 1e-6):
                return False
        return True

    if not is_valid(inner_hex_data, outer_hex_data, outer_hex_side_length):
        # Fallback to a known valid packing (the seed one, or a slightly improved static one)
        # Let's use the seed packing logic but optimized for side
        # Seed:
        inner_fallback = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_fallback = np.array([0, 0, 0])
        side_fallback = 8.0
        return inner_fallback, outer_fallback, side_fallback

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
