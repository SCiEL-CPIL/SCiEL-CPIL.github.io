import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 11
    sqrt3 = np.sqrt(3)
    
    def get_hex_vertices(cx, cy, rot_deg=0.0):
        rot = np.deg2rad(rot_deg)
        angles = np.arange(6) * np.pi / 3 + rot
        return np.array([cx + np.cos(angles), cy + np.sin(angles)]).T

    def min_hex_side_centered(points, theta_norm_deg):
        """
        theta_norm_deg: Angle of the normals to the sides.
        Returns minimal side length S for a hexagon centered at origin 
        with normals at theta_norm_deg + k*60 to contain all points.
        """
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        theta = np.deg2rad(theta_norm_deg)
        angles = theta + np.arange(6) * np.pi / 3
        # Projections of centered points onto normal directions
        projs = np.abs(centered[:, 0] * np.cos(angles) + centered[:, 1] * np.sin(angles))
        max_proj = np.max(projs)
        return 2 * max_proj / sqrt3, centroid

    # Candidate clusters (axial coordinates)
    # Flat-top hexagons: x = sqrt(3)*(q + r/2), y = 1.5*r
    
    # Cluster 1: 7 + 4 "L" shape
    axial_1 = [
        (0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,0), (2,-1), (1,-2), (0,-2)
    ]
    
    # Cluster 2: 7 + 4 "Diamond"
    axial_2 = [
        (0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,0), (-2,0), (0,2), (0,-2)
    ]
    
    # Cluster 3: Linear (Fallback/Comparison)
    # We'll generate this in Cartesian directly if needed, but axial is easier.
    # Linear along x-axis: q=k, r=0
    axial_3 = [(k, 0) for k in range(-(N//2), N - N//2)]
    
    candidates = [axial_1, axial_2, axial_3]
    
    best_S = float('inf')
    best_config = None
    best_theta_vertex = 0.0
    best_center = (0.0, 0.0)
    
    for axial in candidates:
        centers = []
        for q, r in axial:
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            centers.append((x, y))
        
        all_points = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy, 0.0)
            all_points.extend(verts)
        all_points = np.array(all_points)
        
        # Optimize normal angle theta_norm from 0 to 60
        steps = 100
        for i in range(steps + 1):
            theta_norm = 60.0 * i / steps
            S, centroid = min_hex_side_centered(all_points, theta_norm)
            if S < best_S:
                best_S = S
                best_config = centers
                best_center = centroid
                # Convert normal angle to vertex angle for return
                # Vertex angle = Normal angle - 30
                best_theta_vertex = (theta_norm - 30.0) % 60.0
                
        if time.time() > deadline:
            break
            
    if best_config is None:
        # Fallback
        centers = [[sqrt3 * k, 0.0] for k in range(-(N//2), N - N//2)]
        best_S = 40.0
        best_center = (0.0, 0.0)
        best_theta_vertex = 0.0
        best_config = centers
        
    inner = [[cx, cy, 0.0] for cx, cy in best_config]
    return inner, best_center, best_S, best_theta_vertex
