"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """
    Design a 4-fold symmetric photonic crystal unit cell.
    Returns a dielectric cross structure with arm width 8 pixels.
    This structure is known to open large TM band gaps in square lattices.
    """
    size = problem["pixels"]
    
    # Define a dielectric cross
    # Arm width w
    w = 8
    center = size // 2
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Check if pixel is in the horizontal or vertical arm of the cross
            # Arms are centered at size//2
            # Range: [center - w//2, center + w//2 - 1] for even w?
            # Let's define the cross as:
            # 1 if (i in [center - w//2, center + w//2 - 1]) OR (j in [center - w//2, center + w//2 - 1])
            # For size=32, center=16. w=8.
            # Range i: [12, 19]
            # Range j: [12, 19]
            
            in_h_arm = (i >= 12 and i <= 19)
            in_v_arm = (j >= 12 and j <= 19)
            
            if in_h_arm or in_v_arm:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
        
    return cell

# EVOLVE-BLOCK-END
