import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # 10s margin
    
    size = problem["pixels"]
    # Design: Four dielectric rods, one in each quadrant
    # Centers at (size/4, size/4), (size/4, 3*size/4), (3*size/4, size/4), (3*size/4, 3*size/4)
    # Radius r = 5.5 pixels (approx 0.17 of cell size)
    
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    centers = [
        (size/4, size/4),
        (size/4, 3*size/4),
        (3*size/4, size/4),
        (3*size/4, 3*size/4)
    ]
    
    r = 5.5
    
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for cx, cy in centers:
                if (x - cx)**2 + (y - cy)**2 < r**2:
                    cell[i][j] = 1
                    break
    
    return cell
