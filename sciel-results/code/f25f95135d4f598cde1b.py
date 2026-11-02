import numpy as np

def construct():
    # 8 points: vertices of a regular cube
    # 6 points: vertices of a regular octahedron
    # Both scaled to lie on the same sphere of radius sqrt(3)
    
    # Cube vertices (side 2, centered at origin)
    cube_vertices = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0]
    ])
    
    # Octahedron vertices
    # To be on the same sphere as cube vertices (radius sqrt(3)),
    # the octahedron vertices (axes intercepts) must be at distance sqrt(3).
    # Standard octahedron vertices are (1,0,0) etc. Radius 1.
    # Scale factor: sqrt(3).
    s = np.sqrt(3.0)
    octa_vertices = np.array([
        [s, 0.0, 0.0],
        [-s, 0.0, 0.0],
        [0.0, s, 0.0],
        [0.0, -s, 0.0],
        [0.0, 0.0, s],
        [0.0, 0.0, -s]
    ])
    
    points = np.vstack((cube_vertices, octa_vertices))
    
    return points
