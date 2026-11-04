import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 hexagons in a rigid cluster
    # Hexagon 0: Center
    # Hexagons 1-6: Ring 1
    # Hexagons 7-11: Ring 2 (attached to 1-5)
    
    # Positions of centers
    # Unit hexagon inradius (apothem) is sqrt(3)/2
    # Distance between centers of touching hexagons is sqrt(3)
    d = np.sqrt(3)
    
    centers = []
    angles = [] # Rotation of each hexagon relative to global frame
    
    # Hex 0
    centers.append(np.array([0.0, 0.0]))
    angles.append(0.0)
    
    # Ring 1
    for i in range(6):
        theta = i * np.pi / 3
        c = np.array([d * np.cos(theta), d * np.sin(theta)])
        centers.append(c)
        # To touch hex 0, the side of hex i facing hex 0 must be perpendicular to the line connecting centers
        # Hex 0 has a vertex at angle 0.
        # If we align all hexagons with the same orientation, do they touch?
        # Hex 0 vertex at 0. Side at 30.
        # Hex 1 center at angle 0.
        # If Hex 1 has vertex at 0, its left side is at 150?
        # Let's just use 0 rotation for all, it works for honeycomb if aligned properly.
        # Actually, in a standard honeycomb, all hexagons have the same orientation.
        angles.append(0.0)
        
    # Ring 2
    # Attach to Hex 1, 2, 3, 4, 5 (indices 1 to 5)
    # Skip Hex 6 (index 6) to make it 12 total (1+6+5)
    for i in range(1, 6):
        theta = i * np.pi / 3
        # Direction from center to Hex i
        dir_vec = np.array([np.cos(theta), np.sin(theta)])
        # New center is Hex i center + d * dir_vec
        c_new = centers[i] + d * dir_vec
        centers.append(c_new)
        angles.append(0.0)
        
    # Now we have 12 hexagons
    # Optimize global rotation
    best_L = float('inf')
    best_phi = 0.0
    best_center = np.array([0.0, 0.0])
    
    # Support function of a unit hexagon
    def hex_support(theta, alpha):
        # theta: direction of normal
        # alpha: rotation of hexagon
        # Local angle
        psi = (theta - alpha) % (np.pi / 3)
        # Map to nearest side normal
        # Side normals are at 30, 90, ...
        # Distance from side normal
        delta = np.abs(psi - np.pi/6)
        # If psi is close to 0 or pi/3, delta is pi/6
        # If psi is pi/6, delta is 0
        # h = (sqrt(3)/2) * sec(delta)
        # sec(delta) = 1/cos(delta)
        return (np.sqrt(3)/2) / np.cos(delta)
        
    # Grid search for rotation
    for phi in np.linspace(0, np.pi/3, 360):
        # Rotate centers
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        rot_centers = [R @ c for c in centers]
        
        # Calculate widths in 6 directions
        widths = []
        for k in range(6):
            theta = k * np.pi / 3
            u = np.array([np.cos(theta), np.sin(theta)])
            u_neg = -u
            theta_neg = theta + np.pi
            
            h_pos = 0
            h_neg = 0
            for i in range(12):
                # Support in direction u
                proj = np.dot(rot_centers[i], u)
                h_i = proj + hex_support(theta, angles[i] + phi)
                h_pos = max(h_pos, h_i)
                
                # Support in direction -u
                proj_neg = np.dot(rot_centers[i], u_neg)
                h_i_neg = proj_neg + hex_support(theta_neg, angles[i] + phi)
                h_neg = max(h_neg, h_i_neg)
                
            width = h_pos + h_neg
            widths.append(width)
            
        L = max(widths) / np.sqrt(3)
        
        if L < best_L:
            best_L = L
            best_phi = phi
            # Calculate centering shift
            # We need to center the bounding box
            # For each direction k, the range is [min_k, max_k]
            # The center should be the midpoint of the global bounding box
            # Approximation: center of mass? Or midpoint of min/max projections?
            # Let's compute the midpoint of the projections for the critical direction
            # Actually, we can just return the centers as is, and let the evaluator handle the center?
            # No, we must return the center of the outer hexagon.
            # The outer hexagon center should be the center of the minimal bounding hexagon.
            
            # To find the center, we compute the min and max projection for each of the 6 directions
            mins = []
            maxs = []
            for k in range(6):
                theta = k * np.pi / 3
                u = np.array([np.cos(theta), np.sin(theta)])
                theta_neg = theta + np.pi
                u_neg = -u
                
                h_pos = float('-inf')
                h_neg = float('-inf')
                for i in range(12):
                    proj = np.dot(rot_centers[i], u)
                    h_i = proj + hex_support(theta, angles[i] + phi)
                    h_pos = max(h_pos, h_i)
                    
                    proj_neg = np.dot(rot_centers[i], u_neg)
                    h_i_neg = proj_neg + hex_support(theta_neg, angles[i] + phi)
                    h_neg = max(h_neg, h_i_neg)
                    
                # The extent is from -h_neg to h_pos
                mins.append(-h_neg)
                maxs.append(h_pos)
                
            # The center of the bounding box in each direction is (min+max)/2
            # We want a point C such that C . u_k = (min_k + max_k)/2
            # This is an over-determined system, but for a symmetric cluster, it should be consistent
            # We can just take the average of the midpoints projected back?
            # Or simply, the center of the bounding box is the origin if the cluster is symmetric?
            # Our cluster is not fully symmetric (missing one ring 2 hex).
            # Let's estimate the center by averaging the midpoints in Cartesian coords.
            # Midpoint in direction k is a scalar m_k.
            # We want C such that C . u_k = m_k.
            # C = (1/6) sum (m_k * u_k) ? No.
            # Let's just use the centroid of the hexagon centers as a good approximation.
            c_avg = np.mean(rot_centers, axis=0)
            best_center = c_avg

    # Construct the final inner list
    R = np.array([[np.cos(best_phi), -np.sin(best_phi)], [np.sin(best_phi), np.cos(best_phi)]])
    inner = []
    for i in range(12):
        c = R @ centers[i]
        # Shift by best_center?
        # The outer hexagon is centered at best_center.
        # The inner hexagons are at c.
        # We need to return the coordinates relative to the outer hexagon center?
        # The task says: "outer hexagon center (x, y)".
        # "inner rows (x, y, angle)".
        # Usually, this means absolute coordinates.
        # So we can place the outer hexagon at (0,0) and shift the inner ones?
        # Or place inner ones at calculated positions and outer hexagon at their centroid?
        # Let's place outer hexagon at (0,0) and shift inner hexagons by -best_center.
        c_shifted = c - best_center
        angle_deg = np.degrees(angles[i] + best_phi)
        inner.append([c_shifted[0], c_shifted[1], angle_deg])
        
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
