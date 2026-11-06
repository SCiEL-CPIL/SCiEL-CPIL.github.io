"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 unit cell of a square-lattice photonic crystal
    that maximizes the TM band gap between bands 4 and 5.
    
    Strategy: Use a high-filling-fraction dielectric background with a central
    air hole (inverse of the standard rod-in-air). This configuration is known
    to support larger photonic band gaps, particularly in the TM polarization,
    due to the strong contrast and the specific boundary conditions imposed by
    the air holes. A square air hole in a square dielectric background maintains
    the required 4-fold symmetry and can optimize the coupling for the specific
    band pair (4 and 5).
    """
    size = problem["pixels"]
    center = (size - 1) / 2.0
    
    # Define a central square air hole.
    # We want the hole to be symmetric.
    # Let's start with a hole of side 8 pixels (indices 12 to 19).
    # This corresponds to a distance from center < 4.0 pixels.
    hole_half_width = 4.0
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            dx = abs(i - center)
            dy = abs(j - center)
            
            # Inside the hole is air (0). Outside is dielectric (1).
            if dx < hole_half_width and dy < hole_half_width:
                row.append(0)
            else:
                row.append(1)
        cell.append(row)
        
    return cell
# EVOLVE-BLOCK-END
