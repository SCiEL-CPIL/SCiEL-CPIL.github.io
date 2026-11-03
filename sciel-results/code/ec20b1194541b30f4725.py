import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate lattice points
    # Axial coordinates (q, r)
    # x = 1.5 * q
    # y = sqrt(3)/2 * (q + 2*r)
    
    points = []
    # We need 12 points.
    # Shell 0: 1 point (dist 0)
    # Shell 1: 6 points (dist 1)
    # Shell 2: 12 points (dist 2)
    
    # We will pick all from Shell 0 and 1 (7 points)
    # and 5 from Shell 2.
    # To ensure they fit in S_c = 3, we pick the 5 from Shell 2 that are "vertex-aligned"
    # which correspond to axial coords (1,1), (0,2), (-1,2), (-2,1), (-2,0)??
    # Let's just generate all and pick the ones with smallest f(x,y)
    
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = 1.5 * q
            y = (np.sqrt(3) / 2) * (q + 2 * r)
            # Calculate required S_c
            term1 = abs(np.sqrt(3)*x + y) / np.sqrt(3)
            term2 = 2 * abs(y) / np.sqrt(3)
            term3 = abs(-np.sqrt(3)*x + y) / np.sqrt(3)
            sc = max(term1, term2, term3)
            candidates.append((sc, q, r, x, y))
            
    # Sort by required S_c
    candidates.sort(key=lambda t: t[0])
    
    # Select 12 points with smallest S_c
    selected = candidates[:N]
    
    inner = []
    max_sc = 0
    for sc, q, r, x, y in selected:
        inner.append([x, y, 0.0])
        if sc > max_sc:
            max_sc = sc
            
    # Outer side length S = max_sc + 1
    side = max_sc + 1.0
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
