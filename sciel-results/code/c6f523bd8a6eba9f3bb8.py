"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """Concentric square rings with alternating dielectric/air for enhanced TM band gap."""
    size = problem["pixels"]
    center = (size - 1) / 2.0
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Chebyshev distance from center, in pixel units
            dx = abs(i - center)
            dy = abs(j - center)
            d = max(dx, dy)
            ring = int(round(d))
            # Alternate: even rings dielectric, odd rings air
            val = 1 if ring % 2 == 0 else 0
            row.append(val)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
