import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 hexagon centers in a 3x4 grid configuration
    # Centered at (0,0)
    # Row 0: y = -1.5
    # Row 1: y = 0.0
    # Row 2: y = 1.5
    # X coordinates adjusted for centering
    
    sqrt3 = np.sqrt(3)
    
    # Original uncentered 3x4 grid
    # R0: -1.5s, -0.5s, 0.5s, 1.5s
    # R1: -1.0s, 0.0s, 1.0s, 2.0s
    # R2: -1.5s, -0.5s, 0.5s, 1.5s
    # Center of mass X was sqrt3 (which is 1.0s)
    # So we subtract 1.0s from all X
    
    x_coords = {
        0: [-2.5 * sqrt3, -1.5 * sqrt3, -0.5 * sqrt3, 0.5 * sqrt3],
        1: [-2.0 * sqrt3, -1.0 * sqrt3, 0.0 * sqrt3, 1.0 * sqrt3],
        2: [-2.5 * sqrt3, -1.5 * sqrt3, -0.5 * sqrt3, 0.5 * sqrt3]
    }
    y_coords = {
        0: [-1.5],
        1: [0.0],
        2: [1.5]
    }
    
    points = []
    for r in range(3):
        for x in x_coords[r]:
            points.append((x, y_coords[r][0]))
            
    # Function to calculate min side length for a given outer hexagon rotation theta
    def min_side_for_theta(theta):
        # The outer hexagon is rotated by theta.
        # The constraints are |x cos(a) + y sin(a)| <= C
        # where a = theta + k*pi/3
        # C = sqrt(3)/2 * (L - 1)
        # We want min L, so min C.
        # C_min = max over all points and all 6 directions of |x cos(a) + y sin(a)|
        
        max_val = 0.0
        for x, y in points:
            for k in range(6):
                angle = theta + k * np.pi / 3.0
                val = abs(x * np.cos(angle) + y * np.sin(angle))
                if val > max_val:
                    max_val = val
        
        # C = max_val
        # C = sqrt(3)/2 * (L - 1)
        # L = 1 + 2*C / sqrt(3)
        L = 1.0 + 2.0 * max_val / sqrt3
        return L

    # Ternary search for optimal theta in [0, pi/6]
    # Due to symmetry, we only need to search 0 to 30 degrees
    lo = 0.0
    hi = np.pi / 6.0
    eps = 1e-7
    
    for _ in range(100): # Sufficient iterations for double precision
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = min_side_for_theta(m1)
        f2 = min_side_for_theta(m2)
        
        if f1 < f2:
            hi = m2
        else:
            lo = m1
            
    theta_opt = (lo + hi) / 2.0
    L_opt = min_side_for_theta(theta_opt)
    
    # Construct the return values
    # Inner hexagons: (x, y, angle)
    # The inner hexagons are aligned with the global axes (angle 0)
    # Note: The problem allows rotation of inner hexagons. 
    # However, since we fixed the cluster orientation and rotated the outer, 
    # the inner hexagons remain at angle 0 relative to the global frame.
    # The outer hexagon is rotated by theta_opt.
    
    inner = []
    for x, y in points:
        inner.append([x, y, 0.0])
        
    # Outer hexagon: center (0,0), side L_opt, rotation theta_opt
    # The return signature is: inner, center, side, angle
    # angle is the rotation of the outer hexagon in degrees? 
    # "outer hexagon center (x, y), side length and rotation"
    # Usually rotation is in degrees in these tasks if not specified, 
    # but the input angle_degrees suggests degrees. 
    # Let's check the seed: `inner = [[4.0 * k, 0.0, 0.0] ...]` -> angle 0.0.
    # The return `angle` corresponds to the outer hexagon rotation.
    # The task says "angle_degrees" for inner. 
    # For outer, it just says "rotation". 
    # Given the input format uses degrees, and standard practice, 
    # I will return the outer rotation in degrees.
    
    outer_angle_deg = np.degrees(theta_opt)
    
    return inner, (0.0, 0.0), L_opt, outer_angle_deg
# EVOLVE-BLOCK-END
