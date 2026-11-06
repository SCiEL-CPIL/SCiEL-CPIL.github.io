"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a 32x32 unit cell with a 2x2 array of dielectric rods.
    This configuration leverages stronger Bragg scattering for higher-order TM gaps.
    """
    size = problem["pixels"]
    # Initialize with air (0)
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Parameters for the 4 rods
    # Centers of the 4 quadrants in a 32x32 grid
    # Quadrant size is 16. Centers at 8 and 24 (0-indexed: 7.5 to 8.5 approx)
    # Let's use integer centers for simplicity or 0.5 offsets for pixel centers.
    # Pixel (i, j) covers [i, i+1] x [j, j+1]. Center is i+0.5.
    # We want the rod centered in the top-left 16x16 block.
    # The block covers x in [0, 16], y in [0, 16]. Center is (8, 8).
    # So pixel centers should be close to (8, 8).
    # Let's define the rod by its center (cx, cy) and radius r.
    
    # Based on knowledge: 2x2 superlattice of rods is better.
    # Optimal radius is typically around 0.15-0.25 of lattice constant.
    # Lattice constant = 32 pixels.
    # r = 0.2 * 32 = 6.4 pixels.
    r = 6.0
    
    centers = [
        (8.0, 8.0),   # Top-Left
        (24.0, 8.0),  # Top-Right
        (8.0, 24.0),  # Bottom-Left
        (24.0, 24.0)  # Bottom-Right
    ]
    
    for i in range(size):
        for j in range(size):
            # Pixel center coordinates
            px = i + 0.5
            py = j + 0.5
            
            # Check if pixel is inside any of the 4 rods
            for cx, cy in centers:
                dx = px - cx
                dy = py - cy
                if dx*dx + dy*dy < r*r:
                    cell[i][j] = 1
                    break
    
    return cell
# EVOLVE-BLOCK-END
