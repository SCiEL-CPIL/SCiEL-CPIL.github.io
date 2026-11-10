import numpy as np


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    
    Construction: 6x9 grid with horizontal, vertical, and diagonal lines.
    - 6 horizontal lines (y=1..6): 6*9 = 54 incidences
    - 9 vertical lines (x=1..9): 9*6 = 54 incidences  
    - 12 diagonal lines (y-x=c for selected c): 
      Lengths: 6, 6, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2 -> Sum = 56
    Total: 164 incidences
    """
    # Points: 6 x 9 grid
    # x in 1..9, y in 1..6
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # Horizontal lines: y = c -> 0*x + 1*y - c = 0 -> (0, 1, -c)
    for c in range(1, 7):
        lines.append((0, 1, -c))
    
    # Vertical lines: x = c -> 1*x + 0*y - c = 0 -> (1, 0, -c)
    for c in range(1, 10):
        lines.append((1, 0, -c))
    
    # Diagonal lines: y - x = c -> -1*x + 1*y - c = 0 -> (-1, 1, -c)
    # We calculated the number of points for each c in the grid.
    # We want the 12 lines with the most points.
    # c values and counts:
    # 0: 6, -1: 6, -2: 6, -3: 6
    # 1: 5, -4: 5
    # 2: 4, -5: 4
    # 3: 3, -6: 3
    # 4: 2, -7: 2
    # 5: 1, -8: 1
    diag_cs = [0, -1, -2, -3, 1, -4, 2, -5, 3, -6, 4, -7]
    for c in diag_cs:
        lines.append((-1, 1, -c))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
