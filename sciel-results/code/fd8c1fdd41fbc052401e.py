import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    
    # Theoretical minimum side length for 11 unit hexagons is 4.0
    # Configuration:
    # - 1 center hexagon at (0,0), angle 0
    # - 6 hexagons at distance sqrt(3), angles 0,60,120,180,240,300 (touching center)
    # - 4 hexagons at distance 3.0, angles 30,90,150,210 (in gaps of first ring)
    
    sqrt3 = math.sqrt(3.0)
    
    inner = []
    
    # Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for i in range(6):
        angle = i * 60.0
        rad = math.radians(angle)
        x = sqrt3 * math.cos(rad)
        y = sqrt3 * math.sin(rad)
        inner.append([x, y, 0.0])
    
    # Ring 2: 4 hexagons at distance 3.0, in the gaps
    for angle in [30.0, 90.0, 150.0, 210.0]:
        rad = math.radians(angle)
        x = 3.0 * math.cos(rad)
        y = 3.0 * math.sin(rad)
        inner.append([x, y, 0.0])
    
    outer_center = (0.0, 0.0)
    outer_side = 4.0
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
