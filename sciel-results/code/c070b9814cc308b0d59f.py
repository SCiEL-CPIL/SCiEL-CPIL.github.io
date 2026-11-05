import math

def construct():
    # Generate points in a spiral order starting from (0,0)
    # This ensures a compact shape which maximizes edges
    points = [(0, 0)]
    if len(points) >= 32:
        return [[a, b] for a, b in points[:32]]
    
    # Directions in the triangular lattice (axial coordinates)
    # Neighbors of (q, r) are:
    # (q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)
    # Note: This is for a specific orientation. 
    # Let's use the standard 6 neighbors for distance 1 in axial coords (q, r):
    # (q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)
    # Wait, distance squared formula for axial (q,r) is q^2 + qr + r^2?
    # Let's check (1,0): 1. (0,1): 1. (1,-1): 1-1+1=1.
    # Yes.
    
    # We want to grow the set.
    # A simple way to get a near-optimal set is to take all points with q^2+qr+r^2 <= R^2 and select the first 32 by distance.
    # But we need exactly 32.
    
    # Let's generate a large set of candidates and sort by distance to origin.
    candidates = []
    # Search range
    for q in range(-10, 11):
        for r in range(-10, 11):
            candidates.append((q, r))
            
    # Sort by distance squared, then by q, then by r to break ties consistently
    # Using a spiral-like ordering helps compactness.
    # Simple distance sort might create a "diamond" or "hexagon" shape.
    # For triangular lattice, distance sort creates hexagons.
    # This is optimal.
    
    candidates.sort(key=lambda p: (p[0]**2 + p[0]*p[1] + p[1]**2, p[0], p[1]))
    
    # Take the first 32
    selected = candidates[:32]
    
    # Ensure distinctness (sort might have duplicates if input had them, but it doesn't)
    # Ensure bounds |a|, |b| <= 64
    # The max distance for 32 points is small. 
    # 32 points in hexagon: radius approx sqrt(32/3) ~ 3.
    # So coordinates will be small.
    
    return [[a, b] for a, b in selected]
