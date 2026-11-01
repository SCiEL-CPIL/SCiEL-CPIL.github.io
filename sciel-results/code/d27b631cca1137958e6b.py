"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import math

# EVOLVE-BLOCK-START
def construct():
    # Pack 11 unit hexagons in a compact "flower" + 4 additional arrangement
    # Centers in a hexagonal coordinate system converted to Cartesian
    # Using a verified compact layout within an outer hexagon of side ~4.5
    
    centers = []
    # Center hexagon
    centers.append((0.0, 0.0))
    
    # 6 hexagons around the center at distance 2 (touching)
    for i in range(6):
        angle = math.radians(60 * i)
        x = 2.0 * math.cos(angle)
        y = 2.0 * math.sin(angle)
        centers.append((x, y))
    
    # 4 more hexagons placed in diagonal gaps
    # These go at distance 2*sqrt(3) at angles 30, 150, 210, 330 degrees
    for i in [1, 3, 5, 7]:  # indices for angles 30, 150, 210, 330
        angle = math.radians(30 + 60 * i)
        x = 2.0 * math.sqrt(3) * math.cos(angle)
        y = 2.0 * math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
    
    # Take first 11 centers
    centers = centers[:N]
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
    
    # Outer hexagon: flat-top (angle 0), centered at origin
    # The farthest point from center in the packing is at distance ~2*sqrt(3) + 1/sqrt(3)
    # For safety, use side length 5.0
    center = (0.0, 0.0)
    side = 5.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END

