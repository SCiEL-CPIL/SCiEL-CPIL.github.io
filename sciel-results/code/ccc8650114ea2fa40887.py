import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill

# Cache for convex hull information
_HULL_CACHE = {}


def _compute_hull_indices(distance_matrix):
    """
    Compute convex hull indices using a distance-matrix-based approach.
    Returns a set of hull node indices.
    """
    key = id(distance_matrix)
    if key in _HULL_CACHE:
        return _HULL_CACHE[key]
    
    n = distance_matrix.shape[0]
    try:
        # Recover coordinates via MDS
        D2 = distance_matrix ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1][:2]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        coords = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        
        # Use scipy if available for convex hull, otherwise use a simple approach
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            hull_indices = set(hull.vertices)
        except ImportError:
            # Fallback: use distance-based heuristic
            # Nodes that are far from the centroid are likely on the hull
            centroid = np.mean(coords, axis=0)
            dists_to_centroid = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
            threshold = np.percentile(dists_to_centroid, 75)
            hull_indices = set(np.where(dists_to_centroid >= threshold)[0])
        
        _HULL_CACHE[key] = hull_indices
        return hull_indices
    except Exception:
        _HULL_CACHE[key] = set()
        return set()


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Uses a hull-aware heuristic that prefers boundary nodes to reduce
    edge crossings, combined with distance-based scoring.
    """
    # Time guard - if we're running out of time, fall back to greedy
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # For two nodes left, pick the one that minimizes total edge cost to close the tour
        u, v = unvisited_nodes
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    if n_unvisited == 3:
        # For three nodes, evaluate all 6 permutations
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
    
    # Get hull information
    hull_indices = _compute_hull_indices(distance_matrix)
    
    # Distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute hull bonus: nodes on the hull get a bonus to prefer them
    hull_bonus = np.zeros(n_unvisited)
    for i, node in enumerate(unvisited_nodes):
        if node in hull_indices:
            hull_bonus[i] = 1.0
    
    # Adaptive weight for hull preference
    # Stronger hull preference when there are many nodes remaining
    progress = 1.0 - n_unvisited / 50.0
    hull_weight = 0.15 * (1.0 - progress)  # Stronger early in the tour
    
    # Normalize distances for combination
    min_d_curr = np.min(d_curr)
    max_d_curr = np.max(d_curr)
    if max_d_curr > min_d_curr:
        d_curr_norm = (d_curr - min_d_curr) / (max_d_curr - min_d_curr)
    else:
        d_curr_norm = np.zeros(n_unvisited)
    
    min_d_dest = np.min(d_dest)
    max_d_dest = np.max(d_dest)
    if max_d_dest > min_d_dest:
        d_dest_norm = (d_dest - min_d_dest) / (max_d_dest - min_d_dest)
    else:
        d_dest_norm = np.zeros(n_unvisited)
    
    # Combine: primary is distance from current, secondary is destination distance,
    # with a hull bonus that reduces the effective cost of hull nodes
    alpha_dest = 0.1 + 0.2 * progress
    
    scores = d_curr + alpha_dest * d_dest - hull_weight * hull_bonus * (max_d_curr - min_d_curr)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
