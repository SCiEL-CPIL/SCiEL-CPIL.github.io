"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def design(problem):
    """
    Design a 32x32 unit cell with a square-ring (superellipse annulus) structure.
    This shape is known to support larger higher-order TM band gaps compared to simple circular rods
    due to better field confinement and symmetry breaking of the circular mode degeneracy.
    """
    size = problem["pixels"]
    cell = []
    
    # Parameters for the superellipse ring
    # n controls the "squareness" (n=2 is circle, n=4 is squircle, n->inf is square)
    n = 4.0
    # Outer radius (in lattice constants, cell size is 1.0)
    # 0.48 ensures the shape almost touches the neighbors, maximizing interaction
    R_out = 0.48
    # Inner radius
    # 0.22 creates a thick ring
    R_in = 0.22
    
    # Pre-compute powers for efficiency if needed, but 32x32 is small
    # Center is at (15.5, 15.5) in pixel indices
    # x, y range from -0.5 to 0.5
    
    for i in range(size):
        row = []
        y = (i + 0.5) / size - 0.5
        y_abs = abs(y)
        y_pow = y_abs ** (2 * n)
        for j in range(size):
            x = (j + 0.5) / size - 0.5
            x_abs = abs(x)
            x_pow = x_abs ** (2 * n)
            
            val = x_pow + y_pow
            
            # Inside outer superellipse?
            if val < (R_out ** n):
                # Outside inner superellipse?
                if val > (R_in ** n):
                    row.append(1)
                else:
                    row.append(0)
            else:
                row.append(0)
        cell.append(row)
        
    return cell
# EVOLVE-BLOCK-END
