import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Construct a "flower" of 7 hexagons plus 5 outer hexagons
    # Total 12 hexagons
    # All rotated 0 degrees (flat-top or pointy-top depending on definition, 
    # but relative orientation is what matters for packing, container is rotated 0)
    
    # Unit hexagon side length 1.
    # Distance between centers of touching hexagons is sqrt(3).
    
    centers = []
    
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = i * math.pi / 3.0
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 5 hexagons attached to the outside of Ring 1
    # We skip the 6th direction (index 5, angle 300 deg) to make it asymmetric but compact
    # Attached to indices 0, 1, 2, 3, 4 of Ring 1
    
    # Helper to get position attached to a ring 1 hexagon
    # The vector from center to ring 1 hexagon is (dx, dy).
    # The outer hexagon is at (dx, dy) + (dx, dy) = 2*(dx, dy) ?
    # No, the distance is sqrt(3). The vector is (sqrt(3)*cos, sqrt(3)*sin).
    # The next step is in the same direction.
    # So center = 2 * (ring1_center)
    
    for i in range(5): # 0 to 4
        angle = i * math.pi / 3.0
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
        
    # Convert to list of [x, y, angle]
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # We determined S=5.0 is safe.
    # Let's try S=4.7 to get a better score.
    # Based on analysis: S >= 4.61.
    side = 4.7
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
