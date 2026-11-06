"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """
    Design a 32x32 unit cell with a 2x2 array of dielectric square pillars.
    Square shapes break rotational symmetry locally, potentially enhancing
    scattering and band gaps compared to circular rods.
    """
    start_time = time.time()
    deadline = 200  # Safety margin before 230s kill
    
    size = problem["pixels"]
    # Initialize with air (0)
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Parameters for the 4 square pillars
    # Centers of the 4 quadrants in a 32x32 grid
    centers = [
        (8.0, 8.0),   # Top-Left
        (24.0, 8.0),  # Top-Right
        (8.0, 24.0),  # Bottom-Left
        (24.0, 24.0)  # Bottom-Right
    ]
    
    # Half-width of the square pillars
    # Area of circle r=6 is ~113. Area of square 10x10 is 100.
    # Let's try hw = 5.0 (width 10)
    hw = 5.0
    
    for i in range(size):
        if time.time() - start_time > deadline:
            break
        for j in range(size):
            # Pixel center coordinates
            px = i + 0.5
            py = j + 0.5
            
            # Check if pixel is inside any of the 4 square pillars
            for cx, cy in centers:
                dx = abs(px - cx)
                dy = abs(py - cy)
                if dx < hw and dy < hw:
                    cell[i][j] = 1
                    break
    
    return cell
# EVOLVE-BLOCK-END
