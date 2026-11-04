import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Analytical solution for 3x4 grid of unit hexagons
    # Outer hexagon side S
    # Constraint: S >= 1.5 * sqrt(3) + 1 (from width)
    # Constraint: S >= 3.0 (from height, which is looser)
    
    S = 1.5 * math.sqrt(3) + 1.0
    
    # Inner hexagons
    # 3 rows, 4 columns
    # y offsets: -sqrt(3), 0, sqrt(3)
    # x offsets: -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    
    inner = []
    sqrt3 = math.sqrt(3)
    y_coords = [-sqrt3, 0.0, sqrt3]
    x_coords = [-1.5 * sqrt3, -0.5 * sqrt3, 0.5 * sqrt3, 1.5 * sqrt3]
    
    for y in y_coords:
        for x in x_coords:
            inner.append([x, y, 0.0])
            
    # Outer hexagon center
    center = (0.0, 0.0)
    
    # Outer hexagon side
    side = S
    
    # Outer hexagon rotation
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
