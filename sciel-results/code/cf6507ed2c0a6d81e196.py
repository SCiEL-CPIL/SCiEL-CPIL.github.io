import numpy as np
from scipy.optimize import minimize_scalar
import time

def hexagon_packing_12():
    start_time = time.time()
    
    # 12 axial coordinates for a compact cluster
    # The seed used:
    # (0, 0)
    # (1, 0), (-1, 0)
    # (0, 1), (0, -1)
    # (1, -1), (-1, 1)
    # (2, -1), (-2, 1)
    # (2, 0), (-2, 0)
    # (1, -2), (-1, 2)
    # This forms a hexagon of side 2 (19 cells) minus the 6 corners?
    # Side 2 hex has 19 cells. Corners are (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2) in axial?
    # Let's check the seed's list.
    # It includes (2, -1), (-2, 1), (2, 0), (-2, 0), (1, -2), (-1, 2).
    # It does NOT include (0, 2), (0, -2), (2, -2), (-2, 2).
    # So it's a specific 12-cell shape.
    
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (2, 0), (-2, 0),
        (1, -2), (-1, 2)
    ]
    
    # Convert to Cartesian
    # q, r -> x, y
    # x = 1.5 * q
    # y = sqrt(3) * (r + q / 2.0)
    points = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        points.append([x, y])
    points = np.array(points)
    
    # Scale to ensure separation
    # Distance between adjacent centers in unit lattice is sqrt(3).
    # We need gap > 1e-6.
    # Gap = (scale - 1) * sqrt(3).
    # scale > 1 + 1e-6/sqrt(3)
    scale = 1.0 + 1e-5 # Safe margin
    points = points * scale
    
    # Center the cluster at origin
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Precompute vertices of the unit hexagons relative to their centers
    # Hexagon side 1, angle 0.
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    hex_angles = np.deg2rad(np.arange(0, 360, 60))
    hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # Function to calculate minimal enclosing hexagon side length s
    # We fix the outer hexagon orientation to 0 (normals at 0, 60, 120, 180, 240, 300).
    # We rotate the inner cluster by theta.
    def calc_s(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        
        # Rotate centers
        rot_centers = points @ rot.T
        
        # Calculate max projection onto the 6 normals of the outer hexagon
        # Outer hexagon normals (outward) for angle 0:
        # Vertices at 0, 60, ... Normals are at 30, 90, ...?
        # Let's check.
        # Regular hexagon with flat top (angle 0 usually means vertex at 0? Or flat edge at 0?)
        # Problem def: "vertices (cx + d cos(a + 60i deg) ...)"
        # If a=0, vertices are at 0, 60, 120...
        # So the "pointy" top/bottom? No, vertex at 0 is on the right.
        # The edges are between vertex i and i+1.
        # Edge normal is perpendicular to the edge.
        # Midpoint of edge 0-1: angle 30 deg.
        # So normals are at 30, 90, 150, 210, 270, 330 degrees.
        
        normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = -np.inf
        
        # For each inner hexagon
        for i in range(12):
            center = rot_centers[i]
            # Vertices of this hexagon (angle 0 relative to global? 
            # The problem allows angle_deg for inner hexagons. 
            # We can set them to 0. 
            # If we rotate the cluster, we can also rotate the individual hexagons 
            # to maintain their orientation relative to the cluster or global.
            # The seed set angle 0. 
            # If we rotate the centers, should we rotate the hexagons too?
            # Yes, to maintain the packing structure.
            # So the vertices are center + R(theta) * hex_vertices.
            # Or simply, the vertices are at center + hex_vertices rotated by theta?
            # Yes.
            
            verts = center + (hex_vertices @ rot.T)
            
            # Project vertices onto normals
            projs = verts @ normals.T
            # Max projection for this hexagon
            hex_max_proj = np.max(projs)
            
            if hex_max_proj > max_proj:
                max_proj = hex_max_proj
        
        # s * sqrt(3)/2 = max_proj
        s_val = max_proj * 2.0 / np.sqrt(3)
        return s_val

    # Optimize theta
    # The shape has 6-fold symmetry?
    # The cluster of 12 is symmetric under 60 deg rotation?
    # Let's check the coordinates.
    # (0,0)
    # (1,0), (-1,0) -> symmetric
    # (0,1), (0,-1) -> symmetric
    # (1,-1), (-1,1) -> symmetric
    # (2,-1), (-2,1) -> symmetric
    # (2,0), (-2,0) -> symmetric
    # (1,-2), (-1,2) -> symmetric
    # Yes, it has 6-fold rotational symmetry.
    # So we only need to search [0, 60 degrees).
    
    def objective(theta):
        return calc_s(theta)

    res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-8})
    theta_opt = res.x
    
    s_opt = objective(theta_opt)
    
    # Time check
    if time.time() - start_time > 600:
        # Should not happen for this simple optimization, but guard just in case
        pass

    # Construct the result
    c = np.cos(theta_opt)
    s = np.sin(theta_opt)
    rot = np.array([[c, -s], [s, c]])
    final_points = points @ rot.T
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_points[i, 0]
        inner_hex_data[i, 1] = final_points[i, 1]
        inner_hex_data[i, 2] = np.rad2deg(theta_opt) # Rotate the hexagons as well
        
    # Outer hexagon centered at 0,0.
    # Angle 0.
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_opt
