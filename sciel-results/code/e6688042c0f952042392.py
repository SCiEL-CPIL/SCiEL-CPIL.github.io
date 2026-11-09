import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 3 columns, 4 rows of hexagons
    # Horizontal spacing: 2.0
    # Vertical spacing: sqrt(3)
    
    x_centers = [-2.0, 0.0, 2.0]
    y_centers = [-1.5 * np.sqrt(3), -0.5 * np.sqrt(3), 0.5 * np.sqrt(3), 1.5 * np.sqrt(3)]
    
    inner = []
    for y in y_centers:
        for x in x_centers:
            inner.append([x, y, 0.0])
            
    # Side length calculated to fit the bounding box of this arrangement
    # Limiting constraint was |sqrt(3)*x - y| <= sqrt(3)*L
    # Max |sqrt(3)*x - y| was 3.5*sqrt(3)
    # So L = 3.5
    side = 3.5
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
