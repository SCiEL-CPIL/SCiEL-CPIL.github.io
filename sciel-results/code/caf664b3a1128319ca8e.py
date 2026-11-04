import numpy as np

def construct():
    # Side length of outer hexagon
    S = 4.0
    
    # Basis vectors for triangular lattice with spacing sqrt(3)
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)
    v1 = np.array([np.sqrt(3), 0.0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    centers = []
    
    # Add the central hexagon
    centers.append(np.array([0.0, 0.0]))
    
    # Add the 6 neighbors (first shell)
    neighbors = [
        v1,
        v2,
        v2 - v1,
        -v1,
        -v2,
        v1 - v2
    ]
    for n in neighbors:
        centers.append(n)
        
    # Add 5 of the 6 points at distance 3 (second shell "midpoints")
    # These are sums of adjacent basis vectors
    dist3_points = [
        v1 + v2,
        2*v2 - v1,
        v2 - 2*v1,
        -v1 - v2,
        v1 - 2*v2,
        2*v1 - v2
    ]
    
    # Take the first 5
    for p in dist3_points[:5]:
        centers.append(p)
        
    # Construct the inner list
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
