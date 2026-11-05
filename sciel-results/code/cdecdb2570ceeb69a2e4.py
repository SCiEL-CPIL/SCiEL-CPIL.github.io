"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell with four dielectric rods in the quadrants.
    This introduces 2x2 sub-periodicity which is known to enhance the TM band gap.
    We generate the top-left 16x16 quadrant and mirror it to ensure exact C4v symmetry.
    """
    size = problem["pixels"]
    half = size // 2
    
    # Parameters for the rod in the top-left quadrant
    # Center of the 16x16 quadrant (indices 0-15) is at 7.5, 7.5
    cx = 7.5
    cy = 7.5
    radius = 5.0
    
    # Initialize the top-left quadrant
    quadrant = [[0 for _ in range(half)] for _ in range(half)]
    
    # Fill the quadrant
    for i in range(half):
        for j in range(half):
            x = i + 0.5
            y = j + 0.5
            if (x - cx)**2 + (y - cy)**2 < radius**2:
                quadrant[i][j] = 1
    
    # Construct the full 32x32 cell by mirroring the quadrant
    # This guarantees symmetry: cell[i][j] == cell[31-i][j] == cell[i][31-j] == cell[31-i][31-j]
    # And since the quadrant is symmetric (rod is centered), cell[i][j] == cell[j][i]
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(half):
        for j in range(half):
            val = quadrant[i][j]
            cell[i][j] = val
            cell[i][size - 1 - j] = val
            cell[size - 1 - i][j] = val
            cell[size - 1 - i][size - 1 - j] = val
            
    return cell
# EVOLVE-BLOCK-END
