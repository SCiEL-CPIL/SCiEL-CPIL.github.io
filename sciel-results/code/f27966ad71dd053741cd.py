import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 3-4-3-2 packing on a honeycomb lattice
    # Spacing: dx = sqrt(3), dy = 1.5
    dx = math.sqrt(3)
    dy = 1.5
    
    # Row definitions: (y_offset, x_offsets_relative_to_center)
    # We will calculate absolute positions and then center the cluster
    rows = [
        (0.0, [-dx, 0.0, dx]),              # 3 hexes
        (dy,  [-1.5*dx, -0.5*dx, 0.5*dx, 1.5*dx]), # 4 hexes
        (2*dy, [-dx, 0.0, dx]),              # 3 hexes
        (3*dy, [-0.5*dx, 0.5*dx])            # 2 hexes
    ]
    
    inner = []
    for y_off, x_offsets in rows:
        for x_off in x_offsets:
            # Initial angle 0.0
            inner.append([x_off, y_off, 0.0])
            
    # Calculate centroid to center the cluster
    cx = sum(p[0] for p in inner) / N
    cy = sum(p[1] for p in inner) / N
    
    # Shift to center at (0,0)
    for i in range(N):
        inner[i][0] -= cx
        inner[i][1] -= cy
        
    # Determine required outer side length
    # Based on analysis, S ~ 3.6 is sufficient
    # Let's use a safe margin
    side = 3.6
    
    # Outer hexagon center (0,0), rotation 0.0
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
