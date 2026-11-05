"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """A super-ellipse (squircle) dielectric rod with exponent 4 for enhanced higher-order TM band gap."""
    size = problem["pixels"]
    n = 4.0  # super-ellipse exponent (4 = squircle, 2 = circle, large = square)
    radius = 0.28  # half-width of the shape
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Super-ellipse: |x/a|^n + |y/b|^n < 1
            val = abs(x / radius) ** n + abs(y / radius) ** n
            row.append(1 if val < 1.0 else 0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
