import numpy as np
from scipy.spatial import Delaunay

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    # Start with inverse distance for all
    heuristic = 1.0 / (distance_matrix + 1e-9)
    # Set diagonal to 0 or small
    np.fill_diagonal(heuristic, 1e-9)
    
    try:
        tri = Delaunay(coordinates)
        edges = set()
        for simplex in tri.simplices:
            i, j, k = simplex
            edges.add((i, j))
            edges.add((j, k))
            edges.add((k, i))
        
        # Create a boolean mask or just iterate
        # To be efficient, we can create a set of edges and then iterate over the matrix?
        # n=51, so 51x51 = 2601 entries. Iterating is fine.
        
        # We want to penalize non-Delaunay edges.
        # Let's create a set of "allowed" edges for quick lookup
        # Or just multiply non-edges by a factor.
        
        # Efficient way: 
        # 1. Identify Delaunay edges.
        # 2. For all other pairs, reduce heuristic.
        
        # Since n is small, we can just build the set and check.
        delaunay_edges = set()
        for i, j in edges:
            delaunay_edges.add((i, j))
            delaunay_edges.add((j, i))
            
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (i, j) not in delaunay_edges:
                    heuristic[i, j] *= 0.01
    except Exception:
        # Fallback
        pass
        
    return heuristic
