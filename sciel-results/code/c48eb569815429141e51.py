import numpy as np
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 1. Define Hex Lattice Points
    # Axial to Cartesian
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    # Core 7 points
    core_axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    core_cart = np.array([axial_to_cartesian(q, r) for q, r in core_axial])
    
    # Ring 2 points (12 points)
    ring2_axial = [
        (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1),
        (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)
    ]
    # Note: Check if (2,-2) is unique and correct.
    # (2,-2) -> dist (2+2+0)/2 = 2. Yes.
    # Are there duplicates?
    # (2,0) and (-2,0) are distinct.
    # (1,1) and (-1,-1) are distinct.
    # (0,2) and (0,-2) are distinct.
    # (-1,2) and (1,-2) are distinct.
    # (-2,2) and (2,-2) are distinct.
    # (-2,1) and (2,-1) are distinct.
    # Total 12 distinct points.
    
    ring2_cart = np.array([axial_to_cartesian(q, r) for q, r in ring2_axial])

    # 2. Precompute Support Function of Unit Hexagon (Angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300
    # h(theta) = cos(delta) where delta is min angle to vertex
    def unit_hex_support(theta):
        # theta in radians
        # Convert to degrees for modulo logic or use radians
        # Period is 60 deg = pi/3
        # Max is at 0.
        # h(t) = cos(t % (pi/3)) is not quite right because of symmetry around 0.
        # Actually, cos is even and periodic.
        # The function is cos(t) for t in [-pi/6, pi/6].
        # So we can do:
        t = theta % (np.pi / 3.0)
        # Map to [-pi/6, pi/6]
        if t > np.pi / 6.0:
            t = t - np.pi / 3.0
        return np.cos(t)

    # 3. Search over combinations of 4 points from Ring 2
    best_S = float('inf')
    best_config = None
    best_alpha = 0.0
    
    # Generate combinations
    indices = list(range(12))
    for combo in itertools.combinations(indices, 4):
        # Build the 11 points
        extra_points = ring2_cart[list(combo)]
        all_points = np.vstack((core_cart, extra_points))
        
        # Center at origin
        center = np.mean(all_points, axis=0)
        all_points = all_points - center
        
        # Sweep alpha (outer rotation)
        # We need to check 3 directions: alpha, alpha+60, alpha+120
        # S(alpha) = 1/sqrt(3) * max(W(alpha), W(alpha+60), W(alpha+120))
        
        # Optimization: We can just sweep alpha from 0 to 60 deg.
        # Step size: 0.1 deg? 600 steps.
        alphas = np.linspace(0, np.pi/3, 600, endpoint=False)
        
        for alpha in alphas:
            # Calculate widths in 3 directions
            # Direction 1: alpha
            # Direction 2: alpha + pi/3
            # Direction 3: alpha + 2*pi/3
            
            # Vectorized width calculation
            # u1 = [cos(alpha), sin(alpha)]
            # u2 = [cos(alpha+pi/3), sin(alpha+pi/3)]
            # u3 = [cos(alpha+2*pi/3), sin(alpha+2*pi/3)]
            
            u1 = np.array([np.cos(alpha), np.sin(alpha)])
            u2 = np.array([np.cos(alpha + np.pi/3), np.sin(alpha + np.pi/3)])
            u3 = np.array([np.cos(alpha + 2*np.pi/3), np.sin(alpha + 2*np.pi/3)])
            
            # Projections
            p1 = all_points @ u1
            p2 = all_points @ u2
            p3 = all_points @ u3
            
            w1 = (np.max(p1) - np.min(p1)) + 2 * unit_hex_support(alpha)
            w2 = (np.max(p2) - np.min(p2)) + 2 * unit_hex_support(alpha + np.pi/3)
            w3 = (np.max(p3) - np.min(p3)) + 2 * unit_hex_support(alpha + 2*np.pi/3)
            
            max_w = max(w1, w2, w3)
            S = max_w / np.sqrt(3)
            
            if S < best_S:
                best_S = S
                best_alpha = alpha
                best_config = all_points.copy()

    # 4. Construct Return Values
    # inner: list of [x, y, angle]
    inner = []
    for p in best_config:
        inner.append([p[0], p[1], 0.0])
        
    # center: (0,0)
    center = (0.0, 0.0)
    
    # side: best_S
    side = best_S
    
    # angle: best_alpha (in degrees)
    angle = np.degrees(best_alpha)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
