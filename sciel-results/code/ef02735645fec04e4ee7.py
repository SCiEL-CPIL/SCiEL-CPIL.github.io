import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 7 hexagons in the center
    # Center at (0,0)
    inner = [[0.0, 0.0, 0.0]]
    
    # 6 neighbors at distance sqrt(3)
    d1 = np.sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3
        x = d1 * np.cos(angle)
        y = d1 * np.sin(angle)
        # Rotate by 30 degrees? 
        # In standard lattice, if center is 0 deg (vertex at 0), 
        # neighbor at 0 deg has flat side facing center?
        # Let's check orientation.
        # If we use 0.0 for all, they are in a lattice.
        inner.append([x, y, 0.0])
        
    # 4 more hexagons at the next shell
    # Distance 2 * sqrt(3)
    d2 = 2 * np.sqrt(3)
    # Place at angles 0, 90, 180, 270? 
    # Hexagonal symmetry suggests 0, 60, 120, 180? 
    # Let's place them at 0, 60, 120, 180 to keep symmetry with the 6 below.
    # Actually, 4 hexagons. 0, 90, 180, 270 breaks 6-fold symmetry.
    # 0, 60, 120, 180 is better? No, 180 is opposite 0.
    # Let's pick 0, 60, 120, 180.
    angles = [0, np.pi/3, 2*np.pi/3, np.pi]
    
    for angle in angles:
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        # Rotate by 30 degrees (pi/6) so flat side faces center
        # 0.0 is vertex at 0. 30 deg is flat side at 0?
        # If vertex is at 0, flat side is at 60.
        # If we rotate by 30, vertex is at 30, flat side is at 0 (facing center? No, normal is 0).
        # Yes, rotate by 30 degrees.
        inner.append([x, y, 30.0])
        
    # Calculate required side length
    # Max radial distance
    # For the 4 outer hexagons, max distance is d2 + apothem
    # Apothem = sqrt(3)/2
    max_r = d2 + np.sqrt(3)/2
    
    # Check if vertices protrude further into the "sides" of the outer hexagon
    # If outer hexagon is pointy-top (vertex at 0), side at 30.
    # Inner hexagon vertex at 30.
    # Dist to vertex = sqrt(7) approx 2.64
    # Dist to side of outer hexagon = S * sqrt(3)/2
    # We need S * sqrt(3)/2 >= sqrt(7)
    # S >= 2 * sqrt(7) / sqrt(3) = 2 * sqrt(7/3) approx 3.05
    # Our S is 4.33. So 4.33 is larger.
    
    return inner, (0.0, 0.0), max_r, 0.0
# EVOLVE-BLOCK-END
