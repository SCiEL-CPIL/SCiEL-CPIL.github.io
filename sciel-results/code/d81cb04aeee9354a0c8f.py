import numpy as np

def squares_in_square10() -> tuple:
    n = 10
    # 3x3 grid of large squares + 1 small square in the corner
    # Maximize sum = 9*s + t, subject to t <= 1 - 3s
    # Optimal is s -> 1/3, t -> 0
    
    # Use a small epsilon to ensure t > 0 and avoid numerical issues
    eps = 1e-6
    s = (1.0 / 3.0) - eps
    t = 1.0 - 3.0 * s
    
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    idx = 0
    # Place 3x3 grid in bottom-left
    for r in range(3):
        for c in range(3):
            cx = (c + 0.5) * s
            cy = (r + 0.5) * s
            centers[idx] = [cx, cy]
            sides[idx] = s
            idx += 1
            
    # Place 1 small square in top-right
    cx = 1.0 - 0.5 * t
    cy = 1.0 - 0.5 * t
    centers[idx] = [cx, cy]
    sides[idx] = t
    
    return centers, sides
