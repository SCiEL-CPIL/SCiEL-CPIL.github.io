"""Rewritten program with a Diamond Air Hole design for TM polarization."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell with a central diamond-shaped air hole in a dielectric background.
    This leverages the TM polarization preference for holes in high-index dielectrics.
    """
    size = problem["pixels"]
    cell = [[1 for _ in range(size)] for _ in range(size)]
    
    # Center coordinates
    cx = (size - 1) / 2.0 # 15.5
    cy = (size - 1) / 2.0 # 15.5
    
    # Radius of the diamond hole
    # Tuned for TM band gap in square lattice with eps=12
    R = 7.5
    
    for i in range(size):
        for j in range(size):
            if abs(i - cx) + abs(j - cy) < R:
                cell[i][j] = 0
                
    return cell
# EVOLVE-BLOCK-END
