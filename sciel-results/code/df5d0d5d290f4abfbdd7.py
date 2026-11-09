import numpy as np

def szemeredi_trotter_incidences():
    # Grid 7x8
    # x in 1..7, y in 1..8
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Remove (1, 8) and (7, 1)
    # Find indices
    # (1, 8) is at x=1, y=8 -> index (0, 7) in meshgrid -> ravel index 0*8 + 7 = 7
    # (7, 1) is at x=7, y=1 -> index (6, 0) in meshgrid -> ravel index 6*8 + 0 = 48
    # Note: meshgrid with indexing='ij' produces shape (7, 8).
    # ravel order is C-order (row-major).
    # px[i, j] = xs[i], py[i, j] = ys[j].
    # Point (xs[i], ys[j]) is at index i*8 + j.
    # (1, 8): i=0, j=7 -> idx 7.
    # (7, 1): i=6, j=0 -> idx 48.
    
    mask = np.ones(len(points), dtype=bool)
    mask[7] = False
    mask[48] = False
    points = points[mask]
    
    # Lines
    lines = []
    # Verticals x = i -> 1*x + 0*y - i = 0
    for i in range(1, 8):
        lines.append((1, 0, -i))
    # Horizontals y = j -> 0*x + 1*y - j = 0
    for j in range(1, 9):
        lines.append((0, 1, -j))
    # Diagonals y = x + c -> 1*x - 1*y - c = 0
    cs = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6]
    for c in cs:
        lines.append((1, -1, -c))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
