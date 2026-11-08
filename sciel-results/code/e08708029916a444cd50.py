import numpy as np

def construct():
    # N = 11
    # Unit hexagons side 1.
    # Lattice spacing sqrt(3).
    # Centers arranged in a compact cluster.
    # 7 in core (hex number 7), 4 in next layer.
    
    # Core centers (7)
    # (0,0)
    # 6 neighbors at distance sqrt(3)
    d = np.sqrt(3)
    
    centers = [
        [0.0, 0.0],
        [d, 0.0],
        [d/2, 1.5],
        [-d/2, 1.5],
        [-d, 0.0],
        [-d/2, -1.5],
        [d/2, -1.5]
    ]
    
    # Add 4 from Layer 2.
    # We choose 4 points that require S' <= 3.
    # These are the "side" points, not the "corner" points.
    # (0, 3), (0, -3), (3d/2, 1.5), (3d/2, -1.5)
    # Or symmetric: (0,3), (0,-3), (-3d/2, 1.5), (-3d/2, -1.5)
    # Let's pick a symmetric set if possible, or just any 4.
    # To be safe, pick 4 that are well distributed.
    # (0, 3)
    # (0, -3)
    # (3d/2, 1.5)
    # (-3d/2, -1.5) ? No, let's stick to the calculated cheap ones.
    
    p1 = [0.0, 3.0]
    p2 = [0.0, -3.0]
    p3 = [1.5 * d, 1.5]
    p4 = [-1.5 * d, -1.5]
    
    # Check if these are disjoint from core and each other.
    # p1 (0,3) to core (0,0) dist 3 > sqrt(3). OK.
    # p1 to (d/2, 1.5) dist sqrt((d/2)^2 + 1.5^2) = sqrt(3/4 + 2.25) = sqrt(3). OK (touching).
    # p1 to (-d/2, 1.5) dist sqrt(3). OK.
    # p1 to p2 (0,-3) dist 6. OK.
    # p3 (1.5d, 1.5) to (d, 0) dist sqrt((0.5d)^2 + 1.5^2) = sqrt(3/4 + 2.25) = sqrt(3). OK.
    # p3 to (d/2, 1.5) dist d. OK.
    # p3 to p1 (0,3) dist sqrt((1.5d)^2 + 1.5^2) = sqrt(6.75 + 2.25) = sqrt(9) = 3. OK.
    
    centers.append(p1)
    centers.append(p2)
    centers.append(p3)
    centers.append(p4)
    
    # Outer hexagon side S = 4.0
    # Center (0,0)
    # Angle 0.0
    
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    return inner, (0.0, 0.0), 4.0, 0.0
