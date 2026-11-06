import math

def construct():
    # Hexagonal distance function
    def hex_dist(q, r):
        return (abs(q) + abs(r) + abs(q + r)) // 2

    # Generate core points (Hexagon of radius 2)
    core = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if hex_dist(q, r) <= 2:
                core.append((q, r))
    
    # Generate boundary points (Ring 3)
    boundary = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if hex_dist(q, r) == 3:
                boundary.append((q, r))
                
    # Sort boundary by angle to pick a contiguous arc
    def angle(p):
        x = p[0] + p[1] / 2.0
        y = p[1] * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    boundary_sorted = sorted(boundary, key=angle)
    
    # We need 32 points total. Core has 19.
    # Select 13 consecutive points from the boundary.
    # Due to symmetry, any 13 consecutive points work.
    # We take the first 13.
    selected_boundary = boundary_sorted[:13]
    
    final_points = core + selected_boundary
    
    # Ensure exactly 32 distinct points
    # (They are distinct by construction: core is dist<=2, boundary is dist==3)
    
    return [[p[0], p[1]] for p in final_points]
