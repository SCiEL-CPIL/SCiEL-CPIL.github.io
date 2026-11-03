import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # We aim for an outer hexagon of side length 3.0
    # This can contain up to 19 unit hexagons in a hexagonal lattice.
    # We need to place 11.
    
    # Generate all valid axial coordinates (q, r) for a hexagon of side S=3
    S = 3
    candidates = []
    for q in range(-(S - 1), S):
        for r in range(-(S - 1), S):
            # Condition for being inside a hexagon of side S (radius S-1 in hex distance)
            if abs(q) <= S - 1 and abs(r) <= S - 1 and abs(q + r) <= S - 1:
                # Convert axial coordinates to Cartesian (x, y) for flat-top hexagons
                # x = 1.5 * q
                # y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = math.sqrt(3) * (r + q / 2.0)
                # Calculate distance from center to prioritize central placement
                dist = math.sqrt(x * x + y * y)
                candidates.append((dist, x, y))
    
    # Sort by distance from center and take the first N
    candidates.sort(key=lambda item: item[0])
    selected = candidates[:N]
    
    inner = []
    for _, x, y in selected:
        inner.append([x, y, 0.0])
        
    # Outer hexagon parameters
    outer_center = (0.0, 0.0)
    outer_side = 3.0
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
