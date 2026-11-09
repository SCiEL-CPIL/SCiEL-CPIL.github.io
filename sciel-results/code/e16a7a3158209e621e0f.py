# EVOLVE-BLOCK-START
import numpy as np
import time


def szemeredi_trotter_incidences():
    """
    Constructs a 6x9 grid of points and a set of 27 lines (vertical, horizontal, and diagonal)
    to maximize point-line incidences.
    
    Points: x in {0..5}, y in {0..8}. Total 54 points.
    Lines:
    - 6 Vertical lines (x = const), each with 9 points.
    - 9 Horizontal lines (y = const), each with 6 points.
    - 4 Diagonal lines (slope 1, len 6).
    - 4 Diagonal lines (slope -1, len 6).
    - 2 Diagonal lines (slope 1, len 5).
    - 2 Diagonal lines (slope -1, len 5).
    
    Total Incidences: 6*9 + 9*6 + 4*6 + 4*6 + 2*5 + 2*5 = 54 + 54 + 24 + 24 + 10 + 10 = 176.
    """
    start_time = time.time()
    deadline = 600  # Safety margin
    
    # 1. Generate Points
    # Grid 6 (width) x 9 (height)
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate Lines
    lines = []
    
    # Vertical lines: x = i  =>  1*x + 0*y - i = 0
    for i in range(6):
        lines.append((1, 0, -i))
        
    # Horizontal lines: y = j  =>  0*x + 1*y - j = 0
    for j in range(9):
        lines.append((0, 1, -j))
        
    # Diagonal lines Slope 1: y = x + c  =>  1*x - 1*y - c = 0  (Wait: x - y - c = 0? No. y = x + c => x - y + c = 0? No. x - y = -c => x - y + c = 0 is wrong sign for C if we use standard form Ax+By+C=0.
    # y = x + c => -x + y - c = 0. Or x - y + c = 0?
    # If x=0, y=c. 0 - c + c = 0. Correct.
    # So A=1, B=-1, C=c.
    
    # Slope 1, Length 6: c in {0, 1, 2, 3}
    for c in [0, 1, 2, 3]:
        lines.append((1, -1, c))
        
    # Slope 1, Length 5: c in {4, -1}
    for c in [4, -1]:
        lines.append((1, -1, c))
        
    # Diagonal lines Slope -1: y = -x + c  =>  x + y - c = 0
    # A=1, B=1, C=-c
    
    # Slope -1, Length 6: c in {5, 6, 7, 8}
    for c in [5, 6, 7, 8]:
        lines.append((1, 1, -c))
        
    # Slope -1, Length 5: c in {4, 9}
    for c in [4, 9]:
        lines.append((1, 1, -c))
        
    lines_array = np.array(lines, dtype=np.int64)
    
    # 3. Validation and Polishing (Optional but good practice)
    # Check shapes
    if points.shape != (54, 2):
        raise ValueError("Points shape mismatch")
    if lines_array.shape != (27, 3):
        raise ValueError("Lines shape mismatch")
        
    # Check for duplicates in lines (normalized)
    # The lines generated are distinct.
    # (1,0,-i) for i=0..5 are distinct.
    # (0,1,-j) for j=0..8 are distinct.
    # (1,-1,c) for c in {0,1,2,3,4,-1} are distinct.
    # (1,1,-c) for c in {5,6,7,8,4,9} are distinct.
    # No overlap between these sets because (A,B) pairs are (1,0), (0,1), (1,-1), (1,1).
    
    # Check for NaN/Inf (not possible with int64 construction)
    # Check integrality (guaranteed by construction)
    
    return points, lines_array

# EVOLVE-BLOCK-END
