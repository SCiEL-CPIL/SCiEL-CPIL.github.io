import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0  # Safety margin

    # Helper to get vertices of a hexagon with center (cx, cy), side d, angle a (degrees)
    def get_hex_vertices(cx, cy, d, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        x = cx + d * np.cos(angles)
        y = cy + d * np.sin(angles)
        return np.column_stack((x, y))

    # Define the 12 hexagons in a rigid, non-overlapping configuration
    # Hexagonal lattice coordinates
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
    
    # Center the cluster at origin
    centroid = np.mean(base_points, axis=0)
    base_points = base_points - centroid
    
    # Scale slightly to ensure separation > 1e-6 (since base is touching)
    # Distance between centers is sqrt(3) ~ 1.732.
    # Hexagon "radius" (center to vertex) is 1.
    # Hexagon "apothem" (center to edge) is sqrt(3)/2 ~ 0.866.
    # In the lattice, distance between centers is 2 * apothem = sqrt(3).
    # So they touch at edges.
    # We need gap > 1e-6.
    # Scale factor 1.0 + 1e-5 adds 1e-5 * distance to gap?
    # If we scale positions by (1+eps), the distance between centers becomes sqrt(3)*(1+eps).
    # The gap increases by sqrt(3)*eps.
    # We need sqrt(3)*eps > 1e-6 => eps > 1e-6 / 1.732 ~ 6e-7.
    # Let's use 1e-5 to be safe.
    scale_sep = 1.0 + 1e-5
    base_points_scaled = base_points * scale_sep

    # Objective function
    # params: [theta_c, theta_o, dx, dy]
    # theta_c: rotation of cluster
    # theta_o: rotation of outer hexagon
    # dx, dy: offset of cluster center relative to outer hexagon center (which is at 0,0)
    def objective(params):
        theta_c, theta_o, dx, dy = params
        
        # Rotate and translate cluster centers
        c = np.cos(theta_c)
        s = np.sin(theta_c)
        rot_c = np.array([[c, -s], [s, c]])
        
        centers = base_points_scaled @ rot_c.T
        centers[:, 0] += dx
        centers[:, 1] += dy
        
        # Inner hexagons have angle theta_c
        # Outer hexagon has center (0,0) and angle theta_o
        
        # Calculate minimum s for outer hexagon
        # Outer normals
        a_o_rad = np.deg2rad(theta_o)
        normal_angles_o = a_o_rad + np.deg2rad(30 + np.arange(0, 360, 60))
        normals_o = np.column_stack((np.cos(normal_angles_o), np.sin(normal_angles_o)))
        
        max_proj = 0.0
        
        for i in range(12):
            cx, cy = centers[i]
            a_deg_c = np.rad2deg(theta_c)
            verts = get_hex_vertices(cx, cy, 1.0, a_deg_c)
            
            # Project vertices onto outer normals
            # v is (6, 2), normals_o is (6, 2)
            # We need dot(v, n) for each v and n.
            # projs = verts @ normals_o.T -> (6, 6)
            projs = verts @ normals_o.T
            # We need max absolute projection?
            # The constraint is |(v - 0) . n_k| <= s * sqrt(3) / 2
            # So s >= max |(v . n_k)| * 2 / sqrt(3)
            local_max = np.max(np.abs(projs))
            if local_max > max_proj:
                max_proj = local_max
        
        s_val = max_proj * 2.0 / np.sqrt(3.0)
        return s_val

    # Optimization
    best_s = np.inf
    best_params = None
    
    # Multi-start
    # Symmetry: 60 degrees for outer, 60 degrees for cluster?
    # Actually, the cluster is not fully 6-fold symmetric in its bounding box sense, but the lattice is.
    # Let's try a few starts.
    
    starts = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, np.pi/6, 0.0, 0.0],
        [np.pi/6, 0.0, 0.0, 0.0],
        [np.pi/6, np.pi/6, 0.0, 0.0],
        [0.0, np.pi/3, 0.0, 0.0],
    ]
    
    for x0 in starts:
        if time.time() - start_time > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
        except Exception as e:
            pass

    # Fallback if optimization failed
    if best_params is None:
        best_params = [0.0, 0.0, 0.0, 0.0]
        best_s = objective(best_params)

    theta_c, theta_o, dx, dy = best_params
    
    # Construct final result
    c = np.cos(theta_c)
    s = np.sin(theta_c)
    rot_c = np.array([[c, -s], [s, c]])
    
    final_centers = base_points_scaled @ rot_c.T
    final_centers[:, 0] += dx
    final_centers[:, 1] += dy
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = np.rad2deg(theta_c)
    
    outer_hex_data = np.array([0.0, 0.0, np.rad2deg(theta_o)])
    
    # Recalculate s for consistency
    a_o_rad = np.deg2rad(np.rad2deg(theta_o))
    normal_angles_o = a_o_rad + np.deg2rad(30 + np.arange(0, 360, 60))
    normals_o = np.column_stack((np.cos(normal_angles_o), np.sin(normal_angles_o)))
    
    max_proj = 0.0
    for i in range(12):
        cx, cy = final_centers[i]
        a_deg_c = np.rad2deg(theta_c)
        verts = get_hex_vertices(cx, cy, 1.0, a_deg_c)
        projs = verts @ normals_o.T
        local_max = np.max(np.abs(projs))
        if local_max > max_proj:
            max_proj = local_max
            
    s_final = max_proj * 2.0 / np.sqrt(3.0)
    
    return inner_hex_data, outer_hex_data, s_final

# Run once to test
if __name__ == "__main__":
    inner, outer, s = hexagon_packing_12()
    print(f"Side length: {s}")
    print(f"Score: {1/s}")
