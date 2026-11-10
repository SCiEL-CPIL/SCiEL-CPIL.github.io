"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def design(problem):
    """
    Design a symmetric 32x32 pixel unit cell to maximize the TM band gap 
    between bands 4 and 5.
    
    Strategy:
    The seed used a circular dielectric rod with radius 0.2. 
    Photonic crystal literature indicates that square shapes open larger 
    band gaps than circular shapes for the same lattice constant and 
    filling factor, particularly for the TM polarization in a square lattice.
    
    We implement a square dielectric rod centered in the cell. 
    The side length is chosen to be approximately 0.55 lattice constants, 
    which corresponds to a filling factor of ~0.3, often near the optimal 
    range for opening a substantial TM gap in high-index (eps=12) dielectrics.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s budget
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    
    # Default to a square rod design
    # Side length 0.55 means half-side 0.275
    half_side = 0.275
    
    cell = []
    for i in range(size):
        row = []
        # y coordinate for this pixel center
        y = (i + 0.5) / size - 0.5
        for j in range(size):
            # x coordinate for this pixel center
            x = (j + 0.5) / size - 0.5
            
            # Check if inside the square
            if abs(x) < half_side and abs(y) < half_side:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
        
        # Simple time check, though this loop is very fast
        if time.time() > deadline:
            break
            
    return cell
# EVOLVE-BLOCK-END
