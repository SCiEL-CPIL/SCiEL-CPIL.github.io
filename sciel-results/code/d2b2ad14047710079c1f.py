"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Hexagonal cluster: 1 center + 6 ring1 + 4 ring2 = 11
    # All hexagons same orientation (0 degrees)
    # Unit hexagons: circumradius = 1, inradius = sqrt(3)/2
    # Adjacent hexagon center distance for edge-sharing = sqrt(3)
    
    sqrt3 = math.sqrt(3.0)
    
    # Center
    inner = [[0.0, 0.0, 0.0]]
    
    # Ring 1: 6 hexagons at distance sqrt(3), angles 0, 60, 120, 180, 240, 300 degrees
    for i in range(6):
        angle = i * 60.0
        rad = math.radians(angle)
        x = sqrt3 * math.cos(rad)
        y = sqrt3 * math.sin(rad)
        inner.append([x, y, 0.0])
    
    # Ring 2: 4 hexagons at distance 2*sqrt(3), at angles 0, 120, 240, and 300
    # Choose 4 of the 12 ring-2 positions to minimize max extent
    # Place at vertex directions (0, 60, 120, 180, 240, 300) - pick 4
    ring2_angles = [0.0, 120.0, 240.0, 300.0]
    for angle in ring2_angles:
        rad = math.radians(angle)
        x = 2.0 * sqrt3 * math.cos(rad)
        y = 2.0 * sqrt3 * math.sin(rad)
        inner.append([x, y, 0.0])
    
    # Outer hexagon: center at origin, side length = 2*sqrt(3) + 1
    L = 2.0 * sqrt3 + 1.0
    
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
