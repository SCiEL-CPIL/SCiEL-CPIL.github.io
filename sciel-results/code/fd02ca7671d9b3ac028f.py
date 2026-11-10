import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Use 14 vertices from a regular dodecahedron.
    # The dodecahedron has 20 vertices. Taking 14 of them well-spread out
    # should give a high minimum separation relative to the maximum distance.
    
    # Vertices of a regular dodecahedron centered at origin.
    # Let phi = (1 + sqrt(5)) / 2.
    phi = (1 + np.sqrt(5)) / 2.0
    
    # The 20 vertices can be generated from:
    # (±1, ±1, ±1)
    # (0, ±1/phi, ±phi)
    # (±1/phi, ±phi, 0)
    # (±phi, 0, ±1/phi)
    
    vertices = []
    
    # (±1, ±1, ±1)
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.append([i, j, k])
    
    # (0, ±1/phi, ±phi)
    for j in [-1, 1]:
        for k in [-1, 1]:
            vertices.append([0, j/phi, k*phi])
            vertices.append([j*phi, 0, k/phi])
            vertices.append([k/phi, j*phi, 0])
    
    vertices = np.array(vertices)
    
    # We have 8 + 12 = 20 vertices. We need 14.
    # To maximize the minimum distance among the selected 14, we should pick
    # points that are as far apart as possible. One simple heuristic is to 
    # pick every other vertex in a greedy manner, or just take the first 14 
    # and check if it's symmetric.
    
    # A better approach: use a simple greedy algorithm to select 14 points 
    # from the 20 that maximizes the minimum pairwise distance.
    
    all_points = vertices
    n_total = len(all_points)
    n_select = 14
    
    selected = []
    available = list(range(n_total))
    
    # Start with the first point (arbitrary)
    selected.append(available.pop(0))
    
    for _ in range(n_select - 1):
        best_idx = -1
        best_min_dist = -1
        
        for i in available:
            # Calculate minimum distance from point i to all already selected points
            min_dist = min(np.linalg.norm(all_points[i] - all_points[j]) for j in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        
        selected.append(best_idx)
        available.remove(best_idx)
    
    points = all_points[selected]
    
    # Center and normalize
    points = points - np.mean(points, axis=0)
    norms = np.linalg.norm(points, axis=1)
    points = points / np.max(norms)
    
    return points
