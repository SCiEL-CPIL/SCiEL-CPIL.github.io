"""Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap (bands 4-5)."""
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell.
    Uses a single central dielectric circle in air.
    This is a distinct approach from the 2x2 superlattice, focusing on a
    high-fill-factor monolithic structure to maximize the TM band gap.
    """
    size = problem["pixels"]
    cell = [[0] * size for _ in range(size)]
    
    # Center of the 32x32 cell is (15.5, 15.5)
    cx = 15.5
    cy = 15.5
    
    # Radius of the central circle.
    # Tuning this parameter is critical. A radius of 12.5 gives a fill factor
    # of approx pi * 12.5^2 / 1024 ~ 48%. This is often in the sweet spot
    # for opening a large gap between the 4th and 5th TM bands.
    radius = 12.5
    
    for i in range(size):
        for j in range(size):
            # Calculate distance from center
            # Using (i + 0.5) and (j + 0.5) to represent pixel centers
            dist_sq = (i + 0.5 - cx)**2 + (j + 0.5 - cy)**2
            if dist_sq <= radius**2:
                cell[i][j] = 1
    
    return cell
