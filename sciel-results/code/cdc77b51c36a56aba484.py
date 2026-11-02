import numpy as np

def construct():
    # Define the 3x4 hexagonal cluster
    # Basis vectors for hex grid
    u = np.array([1.0, 0.0])
    v = np.array([0.5, np.sqrt(3.0) / 2.0])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * u + j * v
            centers.append(c)
            
    centers = np.array(centers)
    
    # Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Support function of a unit regular hexagon
    # h(alpha) = (sqrt(3)/2) * sec(delta), where delta is angle to nearest 60 deg multiple
    def h_hex(alpha):
        # Normalize angle to [0, 2*pi)
        alpha = alpha % (2 * np.pi)
        # Find nearest multiple of pi/3
        k = np.round(alpha / (np.pi / 3.0)) * (np.pi / 3.0)
        delta = alpha - k
        # Handle wrap-around if needed, but round handles it well for small ranges
        # sec(delta) = 1/cos(delta)
        return (np.sqrt(3.0) / 2.0) / np.cos(delta)
    
    # Objective function: A(theta)
    # theta is the rotation of the inner cluster relative to the outer hexagon
    def objective(theta):
        # Rotation matrix
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate centers
        rot_centers = centers @ R.T
        
        max_extent = 0.0
        # Outer hexagon normals are at 0, 60, 120, 180, 240, 300 degrees
        for k in range(6):
            angle_n = k * np.pi / 3.0
            n = np.array([np.cos(angle_n), np.sin(angle_n)])
            
            # Max projection of rotated centers onto n
            projections = rot_centers @ n
            max_proj = np.max(projections)
            
            # Support function of unit hexagon in direction n
            # The unit hexagons are rotated by theta as well
            # So their orientation relative to n is (angle_n - theta)
            h_val = h_hex(angle_n - theta)
            
            extent = max_proj + h_val
            if extent > max_extent:
                max_extent = extent
                
        return max_extent

    # Minimize objective(theta) using ternary search on [0, pi/6]
    # The function is periodic and symmetric, so [0, pi/6] is sufficient
    lo = 0.0
    hi = np.pi / 6.0
    for _ in range(100):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if objective(m1) < objective(m2):
            hi = m2
        else:
            lo = m1
            
    theta_opt = (lo + hi) / 2.0
    A_opt = objective(theta_opt)
    
    # Calculate side length S
    # A = S * sqrt(3) / 2  =>  S = 2 * A / sqrt(3)
    S = 2.0 * A_opt / np.sqrt(3.0)
    
    # Construct inner hexagon data
    # Centers are already centered at (0,0)
    # Angle is theta_opt (in degrees)
    angle_deg = np.degrees(theta_opt)
    
    inner = []
    for c in centers:
        inner.append([c[0], c[1], angle_deg])
        
    # Outer hexagon center is (0,0), side is S, angle is 0
    outer_center = (0.0, 0.0)
    outer_side = S
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
