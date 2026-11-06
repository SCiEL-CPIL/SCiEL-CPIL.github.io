"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """Air holes in a high-index dielectric (epsilon=12) square lattice.
    
    Uses a square-shaped air hole (0s) in the center of the cell, which is
    known to open large TM band gaps in square-lattice photonic crystals.
    The structure is symmetric under the full square symmetry group (C4v).
    """
    size = problem["pixels"]
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Center of pixel
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Square air hole: |x| < a and |y| < a
            # a ~ 0.30 lattice constants gives a hole of side 0.6
            a = 0.30
            if abs(x) < a and abs(y) < a:
                row.append(0)  # air hole
            else:
                row.append(1)  # dielectric
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
