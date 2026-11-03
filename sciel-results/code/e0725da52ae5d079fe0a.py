import numpy as np

def construct():
    N = 21
    # Hexagonal packing 5, 4, 5, 4, 3
    # r = 2 / (12 + 4*sqrt(3))
    r = 2.0 / (12.0 + 4.0 * np.sqrt(3))
    
    rows = [5, 4, 5, 4, 3]
    circles = []
    
    for i, count in enumerate(rows):
        y = r + i * r * np.sqrt(3)
        # Determine x offset
        # If i is even (0, 2, 4), start at r
        # If i is odd (1, 3), start at 2r
        # Wait, let's verify the offset logic.
        # R0 (i=0, count 5): x = r, 3r, 5r, 7r, 9r. (Start r)
        # R1 (i=1, count 4): x = 2r, 4r, 6r, 8r. (Start 2r)
        # R2 (i=2, count 5): x = r, 3r, 5r, 7r, 9r. (Start r)
        # R3 (i=3, count 4): x = 2r, 4r, 6r, 8r. (Start 2r)
        # R4 (i=4, count 3): x = r, 3r, 5r. (Start r)
        
        if i % 2 == 0:
            start_x = r
        else:
            start_x = 2 * r
            
        for j in range(count):
            x = start_x + j * 2 * r
            circles.append([x, y, r])
            
    return np.array(circles)
