"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def hexagon_vertices_local(side):
        """Returns vertices of a unit hexagon centered at (0,0) with flat top."""
        angles = np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            side * np.cos(angles),
            side * np.sin(angles)
        ])

    def get_cluster_coords(topology_type):
        """
        Returns list of (q, r) axial coordinates for the cluster.
        """
        if topology_type == 0:
            # 3x4 Parallelogram
            return [(q, r) for r in range(3) for q in range(4)]
        elif topology_type == 1:
            # 12 closest to origin
            coords = []
            # Generate candidates
            for r in range(-5, 6):
                for q in range(-5, 6):
                    # Distance squared in hex lattice
                    d2 = q*q + q*r + r*r
                    coords.append((d2, q, r))
            coords.sort()
            return [(q, r) for d2, q, r in coords[:12]]
        elif topology_type == 2:
            # 2x6 Strip
            return [(q, r) for r in range(2) for q in range(6)]
        else:
            return [(q, r) for r in range(3) for q in range(4)]

    def axial_to_xy(q, r):
        """Convert axial coordinates to cartesian for flat-top hexes of side 1."""
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y

    def prepare_cluster(topology_type):
        """
        Pre-compute local vertices of the cluster relative to its centroid.
        Returns array of shape (N*6, 2)
        """
        coords = get_cluster_coords(topology_type)
        # Convert to XY
        pts = np.array([axial_to_xy(q, r) for q, r in coords])
        
        # Center the cluster
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        
        # Generate vertices for each hexagon
        # Each hexagon has 6 vertices
        # We need to transform local hex vertices (centered at 0,0) to each center
        local_hex_verts = hexagon_vertices_local(1.0)
        
        all_verts = []
        for pt in pts:
            verts = local_hex_verts + pt
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts)
        return all_verts

    def check_containment(verts_world, outer_center, outer_side, outer_angle):
        """
        Check if all vertices are inside the outer hexagon.
        Returns penalty (sum of squared distances of points outside).
        """
        # Transform outer hexagon to local frame of vertices?
        # Or transform vertices to local frame of outer hexagon.
        # Let's transform vertices to outer's local frame.
        
        ox, oy = outer_center
        theta = np.radians(outer_angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Translate
        dx = verts_world[:, 0] - ox
        dy = verts_world[:, 1] - oy
        
        # Rotate
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t
        
        # Check against 6 half-planes
        # Apothem = side * sqrt(3) / 2
        apothem = outer_side * np.sqrt(3) / 2.0
        
        # Normal angles for flat-top hexagon (0, 60, 120, 180, 240, 300)
        # Wait, if vertices are at 0, 60..., normals are at 30, 90...?
        # My hexagon_vertices_local uses angles 0, 60...
        # So vertex 0 is at (1, 0).
        # The face opposite to vertex 0 is perpendicular to x-axis.
        # So normals are at 0, 60, 120, 180, 240, 300.
        normal_angles = np.radians(np.arange(0, 360, 60))
        
        # Distances
        # d = local_x * nx + local_y * ny
        # If d > apothem, it's outside.
        
        # Vectorized check
        # local_x shape (M,), local_y shape (M,)
        # We need max over 6 normals for each point?
        # No, a point is inside if it satisfies ALL 6 inequalities.
        # So for each point, we check if ANY inequality is violated.
        
        # Let's compute distance to boundary for each point.
        # dist_i = max(0, max_k (local_x * n_kx + local_y * n_ky - apothem))
        
        # Stack normals
        nxs = np.cos(normal_angles)
        nys = np.sin(normal_angles)
        
        # Compute projections for all points and all normals
        # proj shape (M, 6)
        proj = local_x[:, np.newaxis] * nxs[np.newaxis, :] + local_y[:, np.newaxis] * nys[np.newaxis, :]
        
        # Violation for each point and normal
        viol = np.maximum(0, proj - apothem)
        
        # Max violation per point
        max_viol = np.max(viol, axis=1)
        
        # Sum of squared violations
        penalty = np.sum(max_viol ** 2)
        
        return penalty

    def objective(params, cluster_verts):
        """
        params: [xc, yc, theta_c, theta_o, S]
        cluster_verts: local vertices of the cluster (N*6, 2)
        """
        xc, yc, theta_c, theta_o, S = params
        S = max(S, 0.1)
        
        # Transform cluster to world
        theta_c_rad = np.radians(theta_c)
        cos_c = np.cos(theta_c_rad)
        sin_c = np.sin(theta_c_rad)
        
        local_x = cluster_verts[:, 0]
        local_y = cluster_verts[:, 1]
        
        world_x = local_x * cos_c - local_y * sin_c + xc
        world_y = local_x * sin_c + local_y * cos_c + yc
        
        verts_world = np.column_stack([world_x, world_y])
        
        # Check containment in outer hexagon (center 0,0)
        penalty = check_containment(verts_world, (0, 0), S, theta_o)
        
        return S + penalty * 100.0

    best_score = float('inf')
    best_result = None

    topologies = [0, 1, 2]
    
    for top_idx in topologies:
        if time.time() > deadline - 10:
            break
            
        try:
            cluster_verts = prepare_cluster(top_idx)
            
            # Initial guess
            # Bounding box of cluster
            min_x, min_y = np.min(cluster_verts, axis=0)
            max_x, max_y = np.max(cluster_verts, axis=0)
            width = max_x - min_x
            height = max_y - min_y
            max_dim = max(width, height)
            
            # Estimate S
            # For a hexagon to contain a square of size W, S ~ W / sqrt(3) ?
            # Rough guess
            S_init = max_dim / np.sqrt(3) * 1.5
            
            # Initial params: xc=0, yc=0, theta_c=0, theta_o=0, S=S_init
            x0 = [0, 0, 0, 0, S_init]
            
            # Optimize
            res = minimize(objective, x0, args=(cluster_verts,), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True, 'maxfev': 2000})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = (res.x, top_idx, cluster_verts)
                
        except Exception as e:
            pass

    if best_result is None:
        # Fallback: Return a large hexagon with a valid configuration
        # Use topology 0
        cluster_verts = prepare_cluster(0)
        min_x, min_y = np.min(cluster_verts, axis=0)
        max_x, max_y = np.max(cluster_verts, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        max_dim = max(width, height)
        S = max_dim / np.sqrt(3) * 2.0
        
        # Extract positions
        coords = get_cluster_coords(0)
        pts = np.array([axial_to_xy(q, r) for q, r in coords])
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        
        # Center at 0,0
        inner = []
        for pt in pts:
            inner.append([pt[0], pt[1], 0.0])
        
        return inner, (0, 0), S, 0.0

    # Extract best result
    params, top_idx, cluster_verts = best_result
    xc, yc, theta_c, theta_o, S = params
    
    # Get coordinates
    coords = get_cluster_coords(top_idx)
    pts = np.array([axial_to_xy(q, r) for q, r in coords])
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    
    # Apply rotation and translation
    theta_c_rad = np.radians(theta_c)
    cos_c = np.cos(theta_c_rad)
    sin_c = np.sin(theta_c_rad)
    
    final_pts = []
    for pt in pts:
        x = pt[0] * cos_c - pt[1] * sin_c + xc
        y = pt[0] * sin_c + pt[1] * cos_c + yc
        final_pts.append([x, y, theta_c])
        
    return final_pts, (0, 0), S, theta_o
# EVOLVE-BLOCK-END
