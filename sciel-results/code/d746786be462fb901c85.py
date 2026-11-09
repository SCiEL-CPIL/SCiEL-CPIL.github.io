import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.linalg import eigh

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill

# Cache for coordinates
_COORD_CACHE = {}
# Cache for hull membership per set of unvisited nodes? 
# The set of unvisited nodes changes every call. 
# However, the hull of the *remaining* nodes changes.
# We can cache the full hull of all nodes, but the "active" hull is the hull of the subset.
# Computing ConvexHull on a subset of 50 nodes is fast.

def _recover_coordinates(distance_matrix):
    """
    Recover 2D coordinates from a Euclidean distance matrix using Classical MDS.
    """
    key = id(distance_matrix)
    if key in _COORD_CACHE:
        return _COORD_CACHE[key]
    
    n = distance_matrix.shape[0]
    try:
        # Square the distances
        D2 = distance_matrix ** 2
        
        # Double centering
        # J = I - (1/n) * ones
        ones = np.ones((n, n))
        J = np.eye(n) - ones / n
        
        B = -0.5 * (J @ D2 @ J)
        
        # Eigendecomposition
        # We only need the top 2 eigenvalues/vectors
        vals, vecs = eigh(B)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Take top 2
        # Scale by sqrt of eigenvalues
        coords = vecs[:, :2] * np.sqrt(np.maximum(vals[:2], 0))
        
        # Center the coordinates
        coords -= coords.mean(axis=0)
        
        _COORD_CACHE[key] = coords
        return coords
    except Exception:
        # Fallback: random points if MDS fails
        np.random.seed(42)
        coords = np.random.rand(n, 2)
        _COORD_CACHE[key] = coords
        return coords

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Convex Hull Peeling heuristic.
    1. Recover 2D coordinates.
    2. Compute the convex hull of the unvisited nodes.
    3. If the current node is on the hull (or close to it), prefer the next hull node.
    4. Strategy: Among unvisited nodes, find those on the convex hull of the unvisited set.
       If any exist, pick the one closest to the current node.
       This forces the tour to trace the boundary first.
    5. If no hull nodes remain (all hull nodes visited), pick the closest unvisited node.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Small case: use exact lookahead or simple NN
    if n_unvisited <= 3:
        # For 2 nodes
        if n_unvisited == 2:
            u, v = unvisited_nodes
            cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
            cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
            return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
        # For 3 nodes
        u, v, w = unvisited_nodes
        perms = [
            (u, v, w), (u, w, v),
            (v, u, w), (v, w, u),
            (w, u, v), (w, v, u)
        ]
        best_cost = float('inf')
        best_first = u
        for p in perms:
            c = (distance_matrix[current_node, p[0]] + 
                 distance_matrix[p[0], p[1]] + 
                 distance_matrix[p[1], p[2]] + 
                 distance_matrix[p[2], destination_node])
            if c < best_cost:
                best_cost = c
                best_first = p[0]
        return int(best_first)

    # Get coordinates
    coords = _recover_coordinates(distance_matrix)
    
    # Get coordinates of unvisited nodes
    unvisited_coords = coords[unvisited_nodes]
    
    # Compute convex hull of unvisited nodes
    # Note: ConvexHull requires at least 3 non-collinear points.
    hull_indices = []
    try:
        if n_unvisited >= 3:
            hull = ConvexHull(unvisited_coords)
            hull_indices = hull.vertices
    except Exception:
        # If hull computation fails (e.g., collinear points), treat all as hull or fallback
        hull_indices = list(range(n_unvisited))

    # If we have hull vertices, restrict selection to them
    if len(hull_indices) > 0:
        # Get the actual node IDs for the hull vertices
        hull_node_ids = unvisited_nodes[hull_indices]
        
        # Calculate distances from current node to these hull nodes
        dists = distance_matrix[current_node, hull_node_ids]
        
        # Select the closest hull node
        best_local_idx = np.argmin(dists)
        return int(hull_node_ids[best_local_idx])
    else:
        # No hull (should not happen if n_unvisited >= 3 and non-collinear, 
        # but if all points are collinear or n_unvisited < 3 handled above)
        # Fallback to nearest neighbor among all unvisited
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
