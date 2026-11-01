"""Hexagonal lattice packing of unit hexagons in a minimal enclosing hexagon."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit regular hexagons in a hexagonal lattice pattern,
    then compute the minimal enclosing regular hexagon.
    """
    # Hexagonal lattice basis for unit hexagons (min center-to-center = sqrt(3))
    v1 = np.array([np.sqrt(3.0), 0.0])
    v2 = np.array([np.sqrt(3.0) / 2.0, 1.5])
    
    # Generate candidate lattice points
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            pt = i * v1 + j * v2
            candidates.append((pt, np.linalg.norm(pt)))
    
    # Sort by distance from origin
    candidates.sort(key=lambda x: x[1])
    
    # Select N points starting from origin
    positions = []
    used = set()
    for pt, dist in candidates:
        key = (round(pt[0], 8), round(pt[1], 8))
        if key not in used:
            positions.append(pt)
            used.add(key)
            if len(positions) == N:
                break
    
    pts = np.array(positions)
    
    # Find optimal center: minimize max distance to any point
    # Use the center of the minimum enclosing circle approximation: centroid
    center = np.mean(pts, axis=0)
    
    # For a symmetric hexagonal cluster, the optimal enclosing hexagon is concentric
    # and aligned with the lattice. Compute the required side length.
    
    # The outer hexagon has inradius = S * sqrt(3) / 2
    # Each inner hex has circumradius = 1
    # Condition: for each inner hex center p, the farthest point is at distance |p - center| + 1
    # This must be <= inradius of outer = S * sqrt(3) / 2
    # So S >= 2 * (max_p |p - center| + 1) / sqrt(3)
    
    max_dist = np.max(np.linalg.norm(pts - center, axis=1))
    S = 2.0 * (max_dist + 1.0) / np.sqrt(3.0)
    
    # Build inner list
    inner = [[float(x), float(y), 0.0] for x, y in positions]
    
    return inner, (float(center[0]), float(center[1])), float(S), 0.0
# EVOLVE-BLOCK-END
