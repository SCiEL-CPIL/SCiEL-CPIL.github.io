# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell using a rounded square (super-ellipse) shape.
    This smooths the sharp corners of the previous square design, which can
    reduce high-order scattering and potentially widen the TM band gap.
    """
    size = problem["pixels"]
    # Parameters for the super-ellipse: |x|^n + |y|^n <= R^n
    # n=4 gives a rounded square. R controls the size.
    n = 4.0
    R = 0.29  # Adjusted to provide similar or slightly larger fill fraction than side=0.55 square
    
    cell = []
    for i in range(size):
        y = (i + 0.5) / size
        # Center at 0.5
        dy = y - 0.5
        row = []
        for j in range(size):
            x = (j + 0.5) / size
            dx = x - 0.5
            
            # Super-ellipse equation
            val = abs(dx) ** n + abs(dy) ** n
            if val <= R ** n:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
