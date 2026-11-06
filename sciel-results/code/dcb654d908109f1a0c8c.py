"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # A 16 x 2 strip of the triangular lattice gives 30 edges.
    # We need 76 edges. Let's use a compact hexagonal arrangement.
    # For n=32, the optimal arrangement is a hexagon with side lengths 3,3,3,3,3,3 (36 points) minus 4 points.
    # Actually, let's just try a few known compact shapes.
    
    # A hexagon with radius 3 has 1 + 6 + 12 + 18 = 37 points.
    # If we remove 5 points from the boundary, we might get a good shape.
    
    # Let's generate a hexagon of radius 3 (37 points) and remove 5 boundary points.
    # Hexagon points: (a, b) with max(|a|, |b|, |a+b|) <= 3
    hex_points = [(a, b) for a in range(-3, 4) for b in range(-3, 4) if max(abs(a), abs(b), abs(a+b)) <= 3]
    # There are 37 such points.
    # We need to remove 5.
    
    # Let's try removing 5 corner points to keep it compact.
    # The corners are (3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)
    # Let's remove (3,0), (0,3), (-3,3), (-3,0), (0,-3)
    to_remove = [(3,0), (0,3), (-3,3), (-3,0), (0,-3)]
    points = [p for p in hex_points if p not in to_remove]
    
    # Check if we have 32 points
    if len(points) != 32:
        # Fallback to the 16x2 strip if something goes wrong
        return [[a, b] for a in range(16) for b in range(2)]
    
    # Convert to list of lists
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
