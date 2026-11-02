import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Create a 4x3 grid of hexagon centers
    # Hexagon side length = 1
    # Horizontal distance between centers = sqrt(3)
    # Vertical distance between rows = 1.5
    # Shift between rows = sqrt(3) / 2
    
    dx = math.sqrt(3)
    dy = 1.5
    shift = dx / 2
    
    centers = []
    # 3 rows, 4 columns
    for i in range(3):
        y = (i - 1) * dy
        for j in range(4):
            x = (j - 1.5) * dx + (shift if i % 2 == 1 else 0)
            centers.append([x, y, 0.0])
            
    # Calculate a safe outer side length
    # Based on previous analysis, S ~ 4.5 is safe for this configuration
    # We can try to be slightly tighter if we rotate the outer hexagon
    # But for robustness, we stick to a valid upper bound.
    # Let's use 4.5
    
    inner = centers
    center = [0.0, 0.0]
    side = 4.5
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
