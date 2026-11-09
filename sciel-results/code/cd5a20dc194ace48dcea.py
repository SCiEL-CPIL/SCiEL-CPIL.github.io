import numpy as np

def squares_in_square10() -> tuple:
    """
    Returns a packing of 10 squares in the unit square maximizing the sum of side lengths.
    Uses a strip packing strategy:
    - Top strip (height 0.5): 2 squares of side 0.5.
    - Middle strip (height 0.25): 4 squares of side 0.25.
    - Bottom strip (height 0.25): 4 squares of side 0.25.
    Total sum of sides: 2*0.5 + 8*0.25 = 3.0.
    """
    n = 10
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Top strip: y in [0.5, 1.0]
    # 2 squares of side 0.5
    s_top = 0.5
    centers[0] = [0.25, 0.75]
    sides[0] = s_top
    centers[1] = [0.75, 0.75]
    sides[1] = s_top
    
    # Middle strip: y in [0.25, 0.5]
    # 4 squares of side 0.25
    s_mid = 0.25
    for i in range(4):
        idx = 2 + i
        cx = (i + 0.5) * s_mid
        cy = 0.375
        centers[idx] = [cx, cy]
        sides[idx] = s_mid
        
    # Bottom strip: y in [0.0, 0.25]
    # 4 squares of side 0.25
    s_bot = 0.25
    for i in range(4):
        idx = 6 + i
        cx = (i + 0.5) * s_bot
        cy = 0.125
        centers[idx] = [cx, cy]
        sides[idx] = s_bot
        
    return centers, sides
