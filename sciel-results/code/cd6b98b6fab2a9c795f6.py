import math

N = 12

def construct():
    inner = []
    center = (0.0, 0.0)
    
    # 1. Central hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 2. First layer: 6 hexagons
    # Centers at distance 2 from origin
    # Angles 0, 60, 120, 180, 240, 300 degrees
    for i in range(6):
        angle_rad = i * math.pi / 3
        x = 2.0 * math.cos(angle_rad)
        y = 2.0 * math.sin(angle_rad)
        # Rotation: 0 degrees (flat top or pointy top? 
        # The task says "regular hexagons". Standard is usually flat top or pointy top.
        # The container rotation is independent.
        # Inner hexagons can rotate. 
        # For packing, orientation relative to each other matters.
        # Standard hex grid: all same orientation.
        inner.append([x, y, 0.0])
        
    # 3. Second layer: 5 hexagons
    # Interstitial positions
    # Centers at distance 2*sqrt(3) from origin
    # Angles 30, 90, 150, 210, 270, 330
    # We pick 5 of them. Let's skip 330.
    r2 = 2.0 * math.sqrt(3)
    angles_deg = [30, 90, 150, 210, 270] # Skip 330
    for deg in angles_deg:
        angle_rad = deg * math.pi / 180
        x = r2 * math.cos(angle_rad)
        y = r2 * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length S
    # Max radius of cluster is r2 + 1
    # If container vertex aligns with cluster point, S = r2 + 1
    S = r2 + 1.0
    
    # Container angle
    # Align vertex with 30 degrees
    angle_container = 30.0
    
    return inner, center, S, angle_container
