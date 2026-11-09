import numpy as np

def szemeredi_trotter_incidences():
    """
    Returns a set of 54 points and 27 lines in the integer plane
    that maximizes the number of incidences.
    
    Uses a 6x9 grid of points.
    Points: x in {1..6}, y in {1..9}.
    Lines:
    - 9 horizontal lines (y=c), each with 6 points.
    - 6 vertical lines (x=c), each with 9 points.
    - 12 diagonal lines (y-x=c) with the highest number of points.
    
    Total incidences: 9*6 + 6*9 + (sum of top 12 diagonal lengths)
    Diagonal lengths in 6x9 grid: 6,6,6,6,5,5,4,4,3,3,2,2,1,1.
    Top 12 sum: 52.
    Total: 54 + 54 + 52 = 160.
    """
    # Points: 6x9 grid
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # Horizontal lines: y = c -> 0*x + 1*y - c = 0
    for c in range(1, 10):
        lines.append((0, 1, -c))
        
    # Vertical lines: x = c -> 1*x + 0*y - c = 0
    for c in range(1, 7):
        lines.append((1, 0, -c))
        
    # Diagonal lines: y - x = c -> 1*x - 1*y + c = 0
    # We need the 12 lines with the most points.
    # Counts for c in [-5, 8]:
    # c=-5: 1, -4: 2, -3: 3, -2: 4, -1: 5, 0: 6, 1: 6, 2: 6, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1
    # Top 12: 6,6,6,6,5,5,4,4,3,3,2,2
    # Corresponding c:
    # 6: 0, 1, 2, 3
    # 5: -1, 4
    # 4: -2, 5
    # 3: -3, 6
    # 2: -4, 7
    # (We skip the 1s: -5, 8)
    
    best_c = [0, 1, 2, 3, -1, 4, -2, 5, -3, 6, -4, 7]
    for c in best_c:
        lines.append((1, -1, c))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
