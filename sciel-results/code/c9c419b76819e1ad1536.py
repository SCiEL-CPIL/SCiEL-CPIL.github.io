"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 photonic crystal unit cell.
    Uses a checkerboard pattern with a block size of 4x4 pixels.
    This creates a high-contrast, high-symmetry structure often associated
    with large band gaps in multiple bands.
    """
    size = problem["pixels"]
    block_size = 4
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Determine the block index
            block_i = i // block_size
            block_j = j // block_size
            # Alternate 1 and 0 based on block indices
            if (block_i + block_j) % 2 == 0:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
