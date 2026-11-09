import numpy as np
import time
import itertools

_tour_start_time = None
_instance_cache = {}

def _get_instance_key(distance_matrix):
    """Create a lightweight key for the distance matrix."""
    shape = distance_matrix.shape
    # Sample a few elements to create a hash
    sample = distance_matrix[::10, ::10].flatten()
    if len(sample) == 0:
        sample = distance_matrix.flatten()[:10]
    return (shape, hash(sample.tobytes()))

def _compute_convex_hull_indices(points, unvisited_nodes):
    """
    Compute the convex hull of the unvisited nodes.
    Returns a list of indices into unvisited_nodes that are on the convex hull.
    """
    if len(points) < 3:
        return list(range(len(points)))
    
    # Andrew's monotone chain algorithm
    points_sorted = sorted(enumerate(points), key=lambda x: (x[1][0], x[1][1]))
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for idx, pt in points_sorted:
        while len(lower) >= 2 and cross(points_sorted[lower[-2]][1], points_sorted[lower[-1]][1], pt) <= 0:
            lower.pop()
        lower.append(idx)
        
    upper = []
    for idx, pt in reversed(points_sorted):
        while len(upper) >= 2 and cross(points_sorted[upper[-2]][1], points_sorted[upper[-1]][1], pt) <= 0:
            upper.pop()
        upper.append(idx)
        
    hull_indices = lower[:-1] + upper[:-1]
    # Remove duplicates while preserving order
    seen = set()
    unique_hull = []
    for idx in hull_indices:
        if idx not in seen:
            seen.add(idx)
            unique_hull.append(idx)
            
    return unique_hull

def _solve_exact_tail(current_node, unvisited_nodes, destination_node, distance_matrix):
    """
    Solves the TSP path exactly for the remaining nodes.
    Returns the first node of the optimal path.
    Path: current -> p1 -> p2 -> ... -> pk -> destination
    """
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    
    best_cost = float('inf')
    best_first = int(unvisited_nodes[0])
    
    for perm in itertools.permutations(unvisited_nodes):
        cost = distance_matrix[current_node, perm[0]]
        for i in range(n - 1):
            cost += distance_matrix[perm[i], perm[i+1]]
        cost += distance_matrix[perm[-1], destination_node]
        
        if cost < best_cost:
            best_cost = cost
            best_first = int(perm[0])
            
    return best_first

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _tour_start_time
    
    # Deadline guard
    if _tour_start_time is None:
        _tour_start_time = time.time()
    
    elapsed = time.time() - _tour_start_time
    n_unvisited = len(unvisited_nodes)
    
    # If time is running out, fallback to simple NN
    if elapsed > 115:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Use exact solver for small tails
    if n_unvisited <= 8:
        return _solve_exact_tail(current_node, unvisited_nodes, destination_node, distance_matrix)
    
    # Get coordinates from distance matrix? No, we need coordinates.
    # The interface only provides distance_matrix. 
    # We need to infer coordinates or use a different geometric approach.
    # Wait, the task says "Euclidean instances" but only passes distance_matrix.
    # We can reconstruct coordinates up to rotation/translation/reflection using MDS (Multi-Dimensional Scaling)
    # or we can use the distance matrix directly for geometric heuristics like "angle" which is tricky without coords.
    
    # Let's try to reconstruct coordinates using MDS for the unvisited nodes + current + destination.
    # This is expensive if done every step. 
    # Alternative: Use the distance matrix to estimate "direction" via vector arithmetic in the distance space?
    # Or, simpler: Use a "Geometric" heuristic based on distances to current and destination.
    
    # Heuristic: Minimize d(current, j) + alpha * d(j, destination)
    # But this is just a linear combination.
    
    # Let's try to reconstruct coordinates once per instance.
    key = _get_instance_key(distance_matrix)
    if key not in _instance_cache:
        # Reconstruct coordinates using MDS
        # d^2 = |x_i - x_j|^2
        # We can use sklearn.mds if available, but only numpy/scipy are guaranteed.
        # Let's implement a simple MDS or use scipy.spatial.distance?
        # Actually, we can just use the distance matrix to define a "virtual" coordinate system?
        # No, let's stick to distance-based heuristics that mimic geometric behavior.
        
        # "Convex Hull Peeling" without coordinates:
        # A node is on the "hull" if it is far from the centroid of the other nodes.
        # Or, it has few close neighbors.
        
        # Let's try: Score = d(current, j) + beta * (d(j, destination) - min_d_to_dest)
        # This biases towards nodes that are close to current but not too far from destination?
        
        # Better: "Farthest Point" sampling from the hull?
        
        # Let's try a different structural change: 
        # Use a "Lookahead" with a penalty for "isolated" nodes.
        
        # Or, reconstruct coordinates. 
        # N=50. MDS is O(N^3) or O(N^2 log N).
        # Let's do a simple classical MDS using eigen-decomposition.
        
        D = distance_matrix
        D2 = D ** 2
        J = np.eye(D.shape[0]) - np.ones((D.shape[0], D.shape[0])) / D.shape[0]
        B = -0.5 * J @ D2 @ J
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take first 2 or 3 components
        k = min(3, D.shape[0])
        # Scale by sqrt(eigenvalues)
        coords = eigenvectors[:, :k] * np.sqrt(np.maximum(eigenvalues[:k], 0))
        _instance_cache[key] = coords
    
    coords = _instance_cache[key]
    
    # Now we have coordinates for all nodes.
    # Extract coordinates for current, destination, and unvisited.
    curr_coord = coords[current_node]
    dest_coord = coords[destination_node]
    unvisited_coords = coords[unvisited_nodes]
    
    # Compute convex hull of unvisited nodes
    hull_indices = _compute_convex_hull_indices(unvisited_coords, unvisited_nodes)
    
    # If current node is "outside" or close to hull, prefer hull nodes.
    # Heuristic: 
    # 1. If there are few unvisited nodes, just use NN or exact.
    # 2. Otherwise, score each node.
    #    Score = d(current, j) + lambda * angle_penalty(j)
    #    angle_penalty: How much does the direction current->j deviate from current->dest?
    #    Actually, we want to trace the hull.
    #    So, if j is on the hull, give it a bonus.
    #    Also, prefer nodes that are "ahead" in the hull traversal?
    
    # Let's compute the angle between (current->j) and (current->dest).
    # If the angle is small, j is in the direction of the destination.
    # But we want to visit ALL nodes.
    
    # "Convex Hull Peeling":
    # Prioritize nodes on the convex hull.
    # Among hull nodes, pick the one that is closest to the current node?
    # Or the one that is "next" in the hull order relative to the current node?
    
    # Let's try:
    # 1. Compute distance from current to all unvisited.
    # 2. Compute if node is on hull.
    # 3. Score = d(current, j) - hull_bonus * is_hull(j)
    #    This might pick a far hull node over a close interior node.
    
    # Better: 
    # If the current node is "inside" the hull of unvisited nodes, we are in a "filling" phase.
    # If the current node is "outside" or on the boundary, we are in a "peeling" phase.
    
    # Let's check if current_node is inside the convex hull of unvisited_nodes.
    # This is a bit complex.
    
    # Simpler: 
    # Score = d(current, j) + alpha * d(j, destination)
    # But add a penalty for nodes that are "isolated" (far from other unvisited nodes).
    
    # Let's go back to the "Angle Deviation" idea from the prompt.
    # "select the next node that minimizes the angle deviation from the current direction vector relative to the destination"
    # This implies we have a "current direction". 
    # If we just started, the direction is towards the destination?
    # Or the direction is the vector from the previous node to the current node?
    # We don't have the previous node.
    
    # Let's assume the "current direction" is the vector from the centroid of unvisited nodes to the destination?
    # Or just towards the destination.
    
    # Let's try:
    # Vector v_dest = dest_coord - curr_coord
    # For each j:
    # Vector v_j = unvisited_coords[j] - curr_coord
    # Angle between v_dest and v_j.
    # We want to minimize this angle? That would just pick nodes near the destination.
    
    # Maybe we want to maximize the "progress" towards the destination while staying close?
    
    # Let's try a hybrid:
    # 1. Compute d_current = distance_matrix[current_node, unvisited_nodes]
    # 2. Compute d_dest = distance_matrix[destination_node, unvisited_nodes]
    # 3. Score = d_current + 0.5 * d_dest
    # 4. Among the top 5 candidates, pick the one on the convex hull?
    
    # Let's implement this.
    
    d_current = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    
    scores = d_current + 0.5 * d_dest
    
    # Get top 5 candidates
    top_k = min(5, n_unvisited)
    top_indices = np.argsort(scores)[:top_k]
    
    # Check if any of the top candidates are on the hull
    hull_set = set(hull_indices)
    
    best_node = None
    for idx in top_indices:
        if idx in hull_set:
            best_node = int(unvisited_nodes[idx])
            break
    
    if best_node is None:
        # If no hull node in top 5, pick the best overall
        best_node = int(unvisited_nodes[top_indices[0]])
        
    return best_node
