# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines in the integer plane to maximize
    point-line incidences.
    
    Uses a 6x9 grid of points.
    Points: {(x, y) | 1 <= x <= 6, 1 <= y <= 9}
    
    Lines are chosen to maximize coverage:
    - 6 vertical lines (x = 1..6), each covering 9 points.
    - 9 horizontal lines (y = 1..9), each covering 6 points.
    - 6 diagonal lines of slope 1 (y = x + b) with b in {-1, 0, 1, 2, 3, 4}.
      - b in {0, 1, 2, 3} cover 6 points each.
      - b in {-1, 4} cover 5 points each.
    - 6 diagonal lines of slope -1 (y = -x + b) with b in {6, 7, 8, 9, 10, 11}.
      - b in {7, 8, 9, 10} cover 6 points each.
      - b in {6, 11} cover 5 points each.
      
    Total incidences:
    - Verticals: 6 * 9 = 54
    - Horizontals: 9 * 6 = 54
    - Diag 1: 4 * 6 + 2 * 5 = 24 + 10 = 34
    - Diag -1: 4 * 6 + 2 * 5 = 24 + 10 = 34
    Total I = 54 + 54 + 34 + 34 = 176.
    """
    # Points: 6x9 grid
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []

    # 1. Vertical lines: x = i -> 1*x + 0*y - i = 0
    for i in range(1, 7):
        lines.append((1, 0, -i))

    # 2. Horizontal lines: y = j -> 0*x + 1*y - j = 0
    for j in range(1, 10):
        lines.append((0, 1, -j))

    # 3. Diagonal lines slope 1: y = x + b -> 1*x - 1*y + b = 0
    # b values chosen for max coverage: -1, 0, 1, 2, 3, 4
    for b in [-1, 0, 1, 2, 3, 4]:
        lines.append((1, -1, b))

    # 4. Diagonal lines slope -1: y = -x + b -> 1*x + 1*y - b = 0
    # b values chosen for max coverage: 6, 7, 8, 9, 10, 11
    for b in [6, 7, 8, 9, 10, 11]:
        lines.append((1, 1, -b))

    lines_array = np.array(lines, dtype=np.int64)

    return points, lines_array


# EVOLVE-BLOCK-END
