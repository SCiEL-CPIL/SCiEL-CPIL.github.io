# EVOLVE-BLOCK-START
import numpy as np
import time


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines on the integer lattice.
    Uses a 9x6 grid configuration which yields 176 incidences, a significant improvement
    over the baseline grid (81 incidences).
    """
    start_time = time.time()
    deadline = 600  # seconds

    # 1. Define the 54 points: 9x6 grid
    # x in {1, ..., 9}, y in {1, ..., 6}
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    # 2. Define the 27 lines
    # We select lines with the highest number of incidences.
    # 
    # Group 1: Horizontal lines (6 lines, 9 points each)
    # y = j  =>  0*x + 1*y - j = 0
    lines = []
    for j in range(1, 7):
        lines.append((0, 1, -j))
        
    # Group 2: Vertical lines (9 lines, 6 points each)
    # x = i  =>  1*x + 0*y - i = 0
    for i in range(1, 10):
        lines.append((1, 0, -i))
        
    # Group 3: Diagonal lines x - y = c (4 lines with 6 points)
    # c in {0, 1, 2, 3}
    # x - y - c = 0
    for c in [0, 1, 2, 3]:
        lines.append((1, -1, -c))
        
    # Group 4: Anti-diagonal lines x + y = c (4 lines with 6 points)
    # c in {7, 8, 9, 10}
    # x + y - c = 0
    for c in [7, 8, 9, 10]:
        lines.append((1, 1, -c))
        
    # Group 5: Slope 1/2 lines x - 2y = c (2 lines with 5 points)
    # c in {-1, -3}
    # x - 2y - c = 0
    for c in [-1, -3]:
        lines.append((1, -2, -c))
        
    # Group 6: Slope -1/2 lines x + 2y = c (2 lines with 5 points)
    # c in {11, 13}
    # x + 2y - c = 0
    for c in [11, 13]:
        lines.append((1, 2, -c))
        
    # Total lines: 6 + 9 + 4 + 4 + 2 + 2 = 27
    lines = np.array(lines, dtype=np.int64)

    # Verify count
    if points.shape != (54, 2) or lines.shape != (27, 3):
        raise ValueError("Invalid shapes")

    return points, lines


# EVOLVE-BLOCK-END
