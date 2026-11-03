"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Outer hexagon parameters
    # Side length S = 4.0
    # Rotation 30 degrees (to align vertices with the "easy" lattice points)
    outer_side = 4.0
    outer_center = (0.0, 0.0)
    outer_angle = 30.0
    
    inner = []
    
    # 1. Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 2. Six neighbors at distance sqrt(3), angles 0, 60, 120, 180, 240, 300
    dist_inner_ring = math.sqrt(3)
    for i in range(6):
        angle_rad = math.radians(i * 60)
        x = dist_inner_ring * math.cos(angle_rad)
        y = dist_inner_ring * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    # 3. Five "easy" points at distance 3, angles 30, 90, 150, 210, 270
    # We omit the one at 330 degrees to maintain the count of 12 (1+6+5)
    # and to ensure they align with the vertices of the rotated outer hexagon.
    dist_outer_ring = 3.0
    angles_outer_ring = [30, 90, 150, 210, 270]
    for angle_deg in angles_outer_ring:
        angle_rad = math.radians(angle_deg)
        x = dist_outer_ring * math.cos(angle_rad)
        y = dist_outer_ring * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
