import math
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    # Gap factor to ensure strict disjointness
    epsilon = 0.005
    
    # Lattice basis vectors for hexagonal packing
    # Distance between centers must be sqrt(3) * (1 + epsilon)
    scale = math.sqrt(3) * (1 + epsilon)
    
    # Generate lattice points (k, m) in axial coordinates
    # Cartesian: x = k + m/2, y = m * sqrt(3) / 2
    # Hex norm N(P) = max(|k + m/2|, |k + m|, |k|)
    # We want points with minimal max N(P)
    
    points = []
    # Ring 0
    points.append((0, 0))
    
    # Ring 1
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    points.extend(ring1)
    
    # We have 7 points. Need 4 more from Ring 2.
    # Ring 2 points satisfy N(P) = 2.
    # Let's pick 4 specific ones.
    # (2, 0), (0, 2), (-2, 0), (0, -2)
    ring2_subset = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    points.extend(ring2_subset)
    
    # Now convert to cartesian and scale
    inner = []
    max_norm = 0.0
    
    for k, m in points:
        x = (k + m / 2.0) * scale
        y = (m * math.sqrt(3) / 2.0) * scale
        
        # Calculate hex norm of the scaled point
        # N(P) = max(|x|, |x + y/sqrt(3)|, |x - y/sqrt(3)|)
        # Note: x + y/sqrt(3) = (k + m/2)*scale + (m/2)*scale = (k+m)*scale
        # x - y/sqrt(3) = (k + m/2)*scale - (m/2)*scale = k*scale
        # x = (k + m/2)*scale
        
        n1 = abs(x)
        n2 = abs((k + m) * scale)
        n3 = abs(k * scale)
        
        norm = max(n1, n2, n3)
        if norm > max_norm:
            max_norm = norm
        
        inner.append([x, y, 0.0])
    
    # Required side length S
    # S - 1 >= max_norm
    S = 1.0 + max_norm
    
    # Check deadline
    if time.time() > deadline - margin:
        # Return best found (which is this)
        pass
        
    return inner, (0.0, 0.0), S, 0.0
