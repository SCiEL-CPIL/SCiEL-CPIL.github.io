"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def design(problem):
    """
    Design a symmetric 32x32 unit cell.
    Uses a central square Dielectric rod in an Air background.
    This standard configuration is known to open band gaps for TM polarization.
    """
    size = problem["pixels"]
    # Target: 32x32
    # Structure: Air background (0) with a central Dielectric (1) square rod.
    # Rod size: 14x14 pixels.
    # Center is 15.5.
    # Indices 9 to 22 inclusive (14 pixels).
    # 31 - 9 = 22. Symmetric.
    
    start = 9
    end = 22
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            if start <= i <= end and start <= j <= end:
                row.append(1)  # Dielectric
            else:
                row.append(0)  # Air
        cell.append(row)
    
    return cell
# EVOLVE-BLOCK-END
