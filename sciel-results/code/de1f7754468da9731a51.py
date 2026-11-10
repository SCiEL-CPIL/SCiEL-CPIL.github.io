import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize
import itertools

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill
    
    # Helper to get vertices of a hexagon with center (cx, cy), side d, angle a (degrees)
    def get_hex_vertices(cx, cy, d, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        x = cx + d * np.cos(angles)
        y = cy + d * np.sin(angles)
        return np.column_stack((x, y))
    
    # Helper to check if a point is inside a hexagon defined by center, side, angle
    # Returns True if inside, False otherwise
    def point_in_hex(px, py, cx, cy, d, a_deg):
        # Transform point to hexagon's local coordinate system
        dx = px - cx
        dy = py - cy
        a_rad = np.deg2rad(a_deg)
        # Rotate back by -a_rad
        x_loc = dx * np.cos(-a_rad) - dy * np.sin(-a_rad)
        y_loc = dx * np.sin(-a_rad) + dy * np.cos(-a_rad)
        
        # Check 6 inequalities
        # Normals are at angles 30, 90, 150, 210, 270, 330 relative to hexagon angle
        # Distance from center to edge (apothem) is d * sqrt(3) / 2
        apothem = d * np.sqrt(3) / 2.0
        
        # Project onto 3 axes (0-180, 60-240, 120-300)
        # Axis 1: angle 90 (normal to top/bottom edges if pointy top? No, vertex at 0 means flat top/bottom? 
        # Let's re-verify geometry.
        # Vertices at 0, 60...
        # Edge between 0 and 60: midpoint at 30. Normal at 30.
        # Edge between 60 and 120: midpoint at 90. Normal at 90.
        # Edge between 120 and 180: midpoint at 150. Normal at 150.
        # So normals are at 30, 90, 150, 210, 270, 330.
        
        normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        projections = x_loc * nx + y_loc * ny
        # All projections must be <= apothem
        return np.all(projections <= apothem + 1e-9)
    
    # Helper to compute the minimum side length s of an outer hexagon (centered at origin, angle 0) 
    # that contains a set of inner hexagons (given as centers, angles, side=1)
    def calc_outer_side(centers, angles_deg):
        # Outer hexagon is fixed at center (0,0), angle 0.
        # We need to find s such that all inner hexagons are inside.
        # This is equivalent to finding the max "reach" of any inner hexagon vertex 
        # in the direction of the outer hexagon's edge normals.
        
        # Outer hexagon normals (outward) at angles 30, 90, 150, 210, 270, 330.
        normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        max_proj = 0.0
        
        for i in range(len(centers)):
            cx, cy = centers[i]
            a_deg = angles_deg[i]
            verts = get_hex_vertices(cx, cy, 1.0, a_deg)
            
            # Project vertices onto normals
            projs = verts @ normals.T
            # The outer hexagon boundary in direction n is at distance s * sqrt(3)/2 from center.
            # So we need max_proj <= s * sqrt(3)/2 for all i, n.
            local_max = np.max(projs)
            if local_max > max_proj:
                max_proj = local_max
        
        # s * sqrt(3)/2 = max_proj
        # s = max_proj * 2 / sqrt(3)
        # Note: max_proj can be negative if cluster is centered, but we take max with 0?
        # Actually, if the cluster is centered at 0, the max projection in opposite directions
        # will be similar. The constraint is that the vertex must be inside.
        # Inside condition: projection <= s * sqrt(3)/2 AND projection >= -s * sqrt(3)/2.
        # Since the hexagon is centrally symmetric, if we center the cluster at 0,
        # the max positive projection will determine s.
        
        return max_proj * 2.0 / np.sqrt(3.0)
    
    # Helper to check separation between two inner hexagons
    # Returns True if they are separated (no overlap), False if they overlap
    def are_separated(hex1, hex2):
        # hex1, hex2 are arrays of 6 vertices (6, 2)
        # Separating Axis Theorem (SAT)
        # Axes are normals to edges of both hexagons
        axes = []
        for i in range(6):
            # Edge 1
            v1 = hex1[i]
            v2 = hex1[(i+1)%6]
            edge = v2 - v1
            # Normal
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            axes.append(n)
            
            # Edge 2
            v1 = hex2[i]
            v2 = hex2[(i+1)%6]
            edge = v2 - v1
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            axes.append(n)
        
        for n in axes:
            # Project hex1
            proj1 = hex1 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            
            # Project hex2
            proj2 = hex2 @ n
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            # If max1 < min2 or max2 < min1, they are separated
            # We require separation > 1e-6? 
            # "two inner hexagons not separated: ... projection intervals more than 1e-6 apart"
            # This means if they are NOT separated, the intervals overlap.
            # So valid packing requires intervals to be "more than 1e-6 apart"?
            # Wait. "two inner hexagons not separated: some edge normal ... must give projection intervals more than 1e-6 apart"
            # This phrasing is tricky.
            # "Invalid: ... two inner hexagons not separated: [condition]"
            # So if the condition holds, it is INVALID?
            # "projection intervals more than 1e-6 apart" -> Gap > 1e-6.
            # If Gap > 1e-6, they are separated.
            # So "not separated" means Gap <= 1e-6 (overlap or touch within tolerance).
            # The invalid condition is: "two inner hexagons not separated".
            # So we need them to be separated.
            # Separated means there exists an axis where the gap is > 0.
            # The tolerance 1e-6 likely applies to the "touching" case.
            # If they touch exactly, is it valid?
            # "non-finite values... two inner hexagons not separated".
            # Usually packing allows touching.
            # Let's assume touching is allowed, but we need to ensure they don't overlap significantly.
            # The check "projection intervals more than 1e-6 apart" being the *invalid* condition?
            # No, "Invalid: ... two inner hexagons not separated: [description of test]".
            # This implies the test defines "not separated".
            # If the test says "intervals more than 1e-6 apart", that means they ARE separated.
            # So if they ARE separated, is it invalid? No, that makes no sense.
            # Let's re-read carefully: "Invalid: ... two inner hexagons not separated: some edge normal of either one (separating axis test) must give projection intervals more than 1e-6 apart"
            # This is likely a poorly phrased "For them to be considered separated (and thus valid), ...".
            # OR it means: "It is invalid if they are not separated. To check if they are not separated, we use SAT. If SAT finds no axis with gap > 1e-6, they are not separated."
            # So, Valid = SAT finds an axis with gap > 1e-6?
            # If gap is 0 (touching), is it valid?
            # If gap must be > 1e-6, then touching (gap=0) is invalid.
            # Let's assume strict separation is required (gap > 1e-6).
            
            gap = max(min2 - max1, min1 - max2)
            if gap > 1e-6:
                return True # Separated
        
        return False # Not separated (overlap or touch)
    
    # Define the 12 hexagons.
    # We will use a parameterized configuration.
    # Let's try the "hexagon of side 2 minus corners" shape but allow optimization.
    # Axial coordinates:
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # (2,-1), (-2,1), (2,0), (-2,0), (1,-2), (-1,2)
    
    # Convert axial to Cartesian (standard hex grid)
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (2, 0), (-2, 0),
        (1, -2), (-1, 2)
    ]
    
    base_points = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r)
        base_points.append([x, y])
    base_points = np.array(base_points)
    
    # Center the cluster
    centroid = np.mean(base_points, axis=0)
    base_points = base_points - centroid
    
    # We want to optimize:
    # 1. Global rotation of the cluster
    # 2. Scale factor (to ensure separation)
    # 3. Possibly individual rotations? (Likely not needed if lattice is rigid)
    # 4. Maybe shear/stretch? (Rigid is a good start)
    
    # Let's assume rigid body rotation and scaling.
    # Variables: [theta, scale]
    # Objective: Minimize s (outer side length)
    
    def objective(params):
        theta, scale = params
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        
        # Apply rotation and scale
        centers = base_points @ rot.T * scale
        
        # Check separation
        # If separation fails, return a large penalty
        hex_verts_list = []
        for i in range(12):
            # Inner hexagons also rotate with the cluster?
            # If we rotate the cluster, do the hexagons rotate?
            # Yes, to maintain the packing geometry.
            # Angle of hexagon i is theta (plus initial 0)
            verts = get_hex_vertices(centers[i, 0], centers[i, 1], 1.0, np.rad2deg(theta))
            hex_verts_list.append(verts)
        
        # Check all pairs
        for i in range(12):
            for j in range(i+1, 12):
                if not are_separated(hex_verts_list[i], hex_verts_list[j]):
                    return 1e6 # Penalty
        
        # Calculate outer side
        angles_deg = np.full(12, np.rad2deg(theta))
        s_val = calc_outer_side(centers, angles_deg)
        
        return s_val
    
    # Initial guess
    # Scale must be > 1 to separate.
    # Distance between centers in unit lattice is sqrt(3).
    # Hexagon width (vertex to vertex) is 2.
    # Hexagon height (flat to flat) is sqrt(3).
    # In the lattice, centers are sqrt(3) apart.
    # If hexagons are aligned with lattice, they touch.
    # We need gap > 1e-6.
    # So scale > 1.
    
    # Let's try to optimize theta and scale.
    # We can fix scale to a small epsilon above 1 if we are sure, but optimization is better.
    
    # Multi-start for theta
    best_s = np.inf
    best_params = None
    
    # Time check
    if time.time() - start_time > deadline:
        # Fallback: just return the base config with small scale
        scale = 1.0 + 1e-5
        centers = base_points * scale
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i, 0] = centers[i, 0]
            inner_hex_data[i, 1] = centers[i, 1]
            inner_hex_data[i, 2] = 0.0
        s_val = calc_outer_side(centers, np.zeros(12))
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, s_val
    
    # Optimize
    # We'll do a grid search over theta first to find a good region, then refine.
    # Or just use minimize with multiple starts.
    
    starts = np.linspace(0, np.pi/3, 10) # 6-fold symmetry
    
    for theta0 in starts:
        # Initial scale: 1.0 + 0.01
        x0 = [theta0, 1.01]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
        except Exception:
            pass
            
        if time.time() - start_time > deadline:
            break
    
    # If no valid solution found, use fallback
    if best_params is None or best_s > 1e5:
        # Use simple scaling
        scale = 1.0 + 1e-5
        theta = 0.0
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        centers = base_points @ rot.T * scale
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i, 0] = centers[i, 0]
            inner_hex_data[i, 1] = centers[i, 1]
            inner_hex_data[i, 2] = 0.0
        s_val = calc_outer_side(centers, np.zeros(12))
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, s_val
    
    theta_opt, scale_opt = best_params
    
    # Construct final result
    c = np.cos(theta_opt)
    s = np.sin(theta_opt)
    rot = np.array([[c, -s], [s, c]])
    final_centers = base_points @ rot.T * scale_opt
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = np.rad2deg(theta_opt)
    
    # Outer hexagon
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Calculate final s
    angles_deg = np.full(12, np.rad2deg(theta_opt))
    s_final = calc_outer_side(final_centers, angles_deg)
    
    return inner_hex_data, outer_hex_data, s_final

# Run once to test
if __name__ == "__main__":
    inner, outer, s = hexagon_packing_12()
    print(f"Side length: {s}")
    print(f"Score: {1/s}")
