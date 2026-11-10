import numpy as np

def szemeredi_trotter_incidences():
    # Grid 6x9
    # Points: x in 1..6, y in 1..9
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # Rows: y = c -> 0*x + 1*y - c = 0
    for c in range(1, 10):
        lines.append((0, 1, -c))
        
    # Columns: x = c -> 1*x + 0*y - c = 0
    for c in range(1, 7):
        lines.append((1, 0, -c))
        
    # Diagonals Slope 1: y - x = c -> -1*x + 1*y - c = 0
    # c in {0, 1, 2, 3, 4, -1}
    for c in [0, 1, 2, 3, 4, -1]:
        lines.append((-1, 1, -c))
        
    # Diagonals Slope -1: y + x = c -> 1*x + 1*y - c = 0
    # c in {7, 8, 9, 10, 11, 6}
    for c in [7, 8, 9, 10, 11, 6]:
        lines.append((1, 1, -c))
        
    # We have 9 + 6 + 6 + 6 = 27 lines.
    # Wait, 9+6=15. 6+6=12. Total 27.
    # I previously identified 14 diagonals, but I only listed 12 here.
    # I dropped Slope 2 and Slope -2.
    # This is valid.
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
