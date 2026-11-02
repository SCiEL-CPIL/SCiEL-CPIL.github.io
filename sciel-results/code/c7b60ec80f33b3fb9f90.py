import numpy as np
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_cluster_centers(shape):
        """
        Generate centers for a honeycomb cluster.
        shape: (rows, cols)
        """
        rows, cols = shape
        centers = []
        dx = SQRT3
        dy = 1.5
        shift = SQRT3 / 2
        
        for r in range(rows):
            for c in range(cols):
                x = c * dx
                if r % 2 == 1:
                    x += shift
                y = r * dy
                centers.append([x, y])
        
        centers = np.array(centers)
        # Center the cluster at origin
        center = centers.mean(axis=0)
        centers = centers - center
        return centers

    def min_outer_side(centers, alpha_rad, theta_out_rad=0.0):
        """
        Calculate the minimum outer side length required to contain
        the cluster of hexagons with centers 'centers', rotated by 'alpha_rad',
        inside an outer hexagon rotated by 'theta_out_rad'.
        """
        # Rotate centers
        cos_a = np.cos(alpha_rad)
        sin_a = np.sin(alpha_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_centers = centers @ rot_matrix.T
        
        # The outer hexagon has side S. Its apothem is S * SQRT3 / 2.
        # The condition for a point p to be inside the outer hexagon is:
        # max_k (n_k . p) <= apothem
        # where n_k are the unit outward normals of the outer hexagon.
        
        # Normals for outer hexagon with rotation theta_out
        # The edges are perpendicular to the normals.
        # For a regular hexagon, the normals are at angles theta_out, theta_out + 60, ...
        # Wait, the normal to the flat side is at angle 0 if the side is vertical?
        # Let's stick to the vertex definition.
        # Vertices of outer hexagon (side 1, rot 0) are at angles 0, 60, 120, 180, 240, 300.
        # The outward normal to the edge between v0 and v1 is at angle 30.
        # So normals are at 30, 90, 150, 210, 270, 330 degrees for theta_out=0.
        
        normal_angles = theta_out_rad + np.radians(30 + np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project rotated centers onto normals
        # We also need to account for the radius of the inner hexagons.
        # The inner hexagons have side 1.
        # The farthest point of an inner hexagon from its center in direction n is
        # the support function of the hexagon in direction n.
        # For a regular hexagon of side 1, the support function h(n) is:
        # If n is along a normal, h = apothem = SQRT3/2.
        # If n is along a vertex direction, h = 1.
        # Generally, h(n) = max_{v in vertices} n . v.
        
        # Since all inner hexagons are rotated by the same alpha, their support function
        # in global direction n is the same for all.
        # Let's calculate the support function of a unit hexagon rotated by alpha
        # in the directions of the outer normals.
        
        # Vertices of a unit hexagon rotated by alpha
        inner_angles = alpha_rad + np.radians(np.arange(0, 360, 60))
        inner_verts = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        
        # Support function s_k = max_i (n_k . v_i)
        # n_k is 2x1, inner_verts is 2x6.
        # We want max over i (inner vertex index) for each k (normal index).
        # projections = normals.T @ inner_verts  -> 6x6 matrix where M[k,i] = n_k . v_i
        projections = normals.T @ inner_verts
        support = projections.max(axis=1) # Shape (6,)
        
        # Now, for each center c_j, the max projection is c_j . n_k + support_k.
        # We need max over j and k of (c_j . n_k + support_k) <= S * SQRT3 / 2.
        
        # c_j . n_k is rotated_centers @ normals.T -> 12x6
        center_projections = rotated_centers @ normals.T
        
        # total_max_projection = max(center_projections + support)
        # support is broadcast to 12x6
        total_projections = center_projections + support[np.newaxis, :]
        max_proj = total_projections.max()
        
        S = max_proj * 2.0 / SQRT3
        return S

    def optimize_cluster(centers):
        # We want to minimize min_outer_side(centers, alpha)
        # alpha is in [0, pi/3) due to symmetry.
        # We can use a fine grid search or scipy minimize.
        # Grid search is robust and fast enough.
        
        # Step 1: Coarse grid
        alphas_coarse = np.linspace(0, np.pi/3, 100)
        best_s = np.inf
        best_alpha = 0
        
        for a in alphas_coarse:
            s = min_outer_side(centers, a)
            if s < best_s:
                best_s = s
                best_alpha = a
        
        # Step 2: Fine grid around best
        step = (np.pi/3) / 100
        alphas_fine = np.linspace(best_alpha - step, best_alpha + step, 100)
        for a in alphas_fine:
            if a < 0 or a >= np.pi/3:
                continue
            s = min_outer_side(centers, a)
            if s < best_s:
                best_s = s
                best_alpha = a
                
        return best_s, best_alpha

    # Candidate clusters
    # 1. 3x4 grid
    # 2. 2x6 grid
    # 3. 4x3 grid (transposed)
    # 4. 6x2 grid (transposed)
    
    clusters = []
    clusters.append(get_cluster_centers((3, 4)))
    clusters.append(get_cluster_centers((2, 6)))
    clusters.append(get_cluster_centers((4, 3)))
    clusters.append(get_cluster_centers((6, 2)))
    
    best_s = np.inf
    best_alpha = 0
    best_centers = None
    
    for centers in clusters:
        if time.time() > deadline - 5:
            break
        s, alpha = optimize_cluster(centers)
        if s < best_s:
            best_s = s
            best_alpha = alpha
            best_centers = centers

    # Fallback if something goes wrong or no cluster found
    if best_centers is None:
        best_centers = get_cluster_centers((3, 4))
        best_s, best_alpha = optimize_cluster(best_centers)

    # Construct the final output
    # The inner hexagons are at best_centers rotated by best_alpha.
    # The outer hexagon is at (0,0), side best_s, rotation 0.
    
    # We need to return:
    # inner: list of (x, y, angle_degrees) for each of the 12 hexagons
    # center: (0, 0)
    # side: best_s
    # angle: 0 (outer rotation)
    
    # The angle of the inner hexagons is best_alpha (in radians) converted to degrees.
    # The positions are the rotated centers.
    
    cos_a = np.cos(best_alpha)
    sin_a = np.sin(best_alpha)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    final_centers = best_centers @ rot_matrix.T
    
    inner = []
    for i in range(N):
        x = final_centers[i, 0]
        y = final_centers[i, 1]
        angle_deg = np.degrees(best_alpha)
        inner.append([x, y, angle_deg])
        
    outer_center = (0.0, 0.0)
    outer_side = best_s
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg
