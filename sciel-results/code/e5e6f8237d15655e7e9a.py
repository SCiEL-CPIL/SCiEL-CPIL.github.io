"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell with a central Air Superellipse (rounded square) 
    in a Dielectric background.
    
    This is a structural change from the seed (Dielectric Circle in Air).
    It inverts the dielectric contrast (Air hole in Dielectric) and changes 
    the shape from circular to superelliptical (n=4), which is known to 
    enhance Bragg scattering in square lattices for higher-order bands.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # Initialize as all Dielectric (1)
    cell = [[1 for _ in range(size)] for _ in range(size)]
    
    # Define the Air Hole (0)
    # Center of the cell
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    
    # Superellipse parameters
    # n=4 gives a rounded square shape
    n = 4.0
    # Radius parameter: We want a hole roughly 12 pixels wide.
    # Max extent is R. So R=6 means width 12.
    R = 6.0
    
    for i in range(size):
        for j in range(size):
            # Coordinates relative to center, scaled by R
            x = (i - cx) / R
            y = (j - cy) / R
            
            # Superellipse equation: |x|^n + |y|^n <= 1
            # We use a slightly relaxed threshold to ensure connectivity and smooth edges
            val = abs(x) ** n + abs(y) ** n
            if val < 1.0:
                cell[i][j] = 0
            else:
                cell[i][j] = 1
                
    return cell
# EVOLVE-BLOCK-END
