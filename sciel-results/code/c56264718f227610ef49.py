import numpy as np

def construct():
    # Define the 13 points from the triangular lattice
    # Lattice vectors: u=(1,0), v=(0.5, sqrt(3)/2)
    # Points (i,j) in lattice coords -> x = i + 0.5j, y = sqrt(3)/2 * j
    
    # Indices (i, j) for the 13 closest points to origin
    # d^2 = i^2 + ij + j^2
    # 1 point at d^2=0: (0,0)
    # 6 points at d^2=1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 6 points at d^2=3: (1,1), (2,-1), (-1,2), (-2,1), (-1,-1), (1,-2)
    
    indices = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (2, -1), (-1, 2), (-2, 1), (-1, -1), (1, -2)
    ]
    
    points = []
    for i, j in indices:
        x = i + 0.5 * j
        y = (np.sqrt(3) / 2) * j
        points.append([x, y])
        
    return np.array(points)
