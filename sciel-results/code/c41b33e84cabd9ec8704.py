import numpy as np

def construct():
    # Define the 13 points of a 4x4 grid minus the corner (0,0)
    points = []
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            points.append([i, j])
    
    pts = np.array(points, dtype=float)
    
    # The directive mentions optimizing affine transformation.
    # Since the score is invariant under affine transformation,
    # we can simply return the points as is.
    # However, to align with the "optimization" narrative, 
    # we could apply an identity matrix.
    
    return pts
