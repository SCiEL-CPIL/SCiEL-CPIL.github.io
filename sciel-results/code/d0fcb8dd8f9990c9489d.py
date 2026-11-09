# EVOLVE-BLOCK-START
import numpy as np
import time


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for n=54, m=27.
    Uses a 6x9 grid point set and selects lines to maximize incidences.
    Points: 6x9 grid (54 points).
    Lines: 
      - 6 horizontal lines (x = const)
      - 9 vertical lines (y = const)
      - 12 diagonal lines (y = x + c) with longest segments
    Total lines: 6 + 9 + 12 = 27.
    """
    # Points: 6x9 grid
    # x in 1..6, y in 1..9
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []

    # 6 Horizontal lines: x = i  => 1*x + 0*y - i = 0
    for i in range(1, 7):
        lines.append((1, 0, -i))

    # 9 Vertical lines: y = j  => 0*x + 1*y - j = 0
    for j in range(1, 10):
        lines.append((0, 1, -j))

    # 12 Diagonal lines: y = x + c  => 1*x - 1*y + c = 0
    # Selected c values based on segment length in 6x9 grid
    # c=0,1,2,3 (len 6)
    # c=-1,4 (len 5)
    # c=-2,5 (len 4)
    # c=-3,6 (len 3)
    # c=-4,7 (len 2)
    c_vals = [0, 1, 2, 3, -1, 4, -2, 5, -3, 6, -4, 7]
    for c in c_vals:
        lines.append((1, -1, c))

    lines = np.array(lines, dtype=np.int64)

    return points, lines


# EVOLVE-BLOCK-END
