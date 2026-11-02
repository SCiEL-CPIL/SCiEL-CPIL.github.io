import math

def construct():
    # Lattice vectors for close-packed hexagons (flat-top)
    # Distance between centers is sqrt(3)
    # To ensure strict disjointness, add a small epsilon to the spacing
    epsilon = 0.001
    sqrt3 = math.sqrt(3.0)
    
    # Basis vectors for the triangular lattice of centers
    # A is vertical, B is 30 degrees from horizontal
    Ax = 0.0
    Ay = sqrt3 + epsilon
    
    Bx = 1.5 + epsilon
    By = sqrt3 / 2.0 + epsilon
    
    # Generate 11 centers using axial coordinates (i, j) -> i*A + j*B
    # This generates a cluster
    axial_coords = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
        (0, 2),
        (0, -2),
        (-1, 2),  # This is (3, 0) approx
        (1, -2)   # This is (-3, 0) approx
    ]
    
    inner = []
    centers = []
    for i, j in axial_coords:
        x = i * Ax + j * Bx
        y = i * Ay + j * By
        centers.append((x, y))
        # Inner hexagons are flat-top (angle 0)
        inner.append([x, y, 0.0])
        
    # Calculate the required outer side length
    # The outer hexagon must contain all vertices of the inner hexagons.
    # Inner hexagon vertices relative to center (flat-top, side 1):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    max_r_sq = 0.0
    h_half = sqrt3 / 2.0
    for cx, cy in centers:
        # Check all 6 vertices
        # 1
        dx = cx + 1.0
        dy = cy
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 2
        dx = cx + 0.5
        dy = cy + h_half
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 3
        dx = cx - 0.5
        dy = cy + h_half
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 4
        dx = cx - 1.0
        dy = cy
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 5
        dx = cx - 0.5
        dy = cy - h_half
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 6
        dx = cx + 0.5
        dy = cy - h_half
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq

    # The circumradius of the outer hexagon must be at least max_r
    # Add a small margin to ensure "inside" strictly
    side = math.sqrt(max_r_sq) + 0.01
    
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, center, side, outer_angle
