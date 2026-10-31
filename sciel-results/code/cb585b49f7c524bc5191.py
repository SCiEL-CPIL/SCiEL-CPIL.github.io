import numpy as np
import time
from collections import defaultdict

_start_time = time.time()
_LIMIT = 115.0

# Quadtree-based spatial partitioning state
_quadtree_cache = {}

def _build_quadtree(points, max_depth=6):
    """
    Build a quadtree from 2D points.
    points: (N, 2) array of coordinates
    Returns: list of leaves, each leaf is (center_x, center_y, width, height, [point_indices])
    """
    if len(points) == 0:
        return []
    
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    # Handle degenerate cases
    if max_x - min_x < 1e-10:
        max_x = min_x + 1e-10
    if max_y - min_y < 1e-10:
        max_y = min_y + 1e-10
    
    def _build_recursive(xmin, xmax, ymin, ymax, indices, depth):
        if depth > max_depth or len(indices) <= 2:
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            return [(cx, cy, w, h, list(indices))]
        
        mid_x = (xmin + xmax) / 2.0
        mid_y = (ymin + ymax) / 2.0
        
        # Partition points into 4 quadrants
        q1 = []  # top-left
        q2 = []  # top-right
        q3 = []  # bottom-left
        q4 = []  # bottom-right
        
        for idx in indices:
            px, py = points[idx]
            if px <= mid_x:
                if py <= mid_y:
                    q1.append(idx)
                else:
                    q3.append(idx)
            else:
                if py <= mid_y:
                    q2.append(idx)
                else:
                    q4.append(idx)
        
        leaves = []
        if q1:
            leaves.extend(_build_recursive(xmin, mid_x, ymin, mid_y, q1, depth + 1))
        if q2:
            leaves.extend(_build_recursive(mid_x, xmax, ymin, mid_y, q2, depth + 1))
        if q3:
            leaves.extend(_build_recursive(xmin, mid_x, mid_y, ymax, q3, depth + 1))
        if q4:
            leaves.extend(_build_recursive(mid_x, xmax, mid_y, ymax, q4, depth + 1))
        
        return leaves
    
    all_indices = list(range(len(points)))
    return _build_recursive(min_x, max_x, min_y, max_y, all_indices, 0)

def _get_quadtree(points):
    """Get or build quadtree for given points."""
    key = points.tobytes()
    if key not in _quadtree_cache:
        _quadtree_cache[key] = _build_quadtree(points)
    return _quadtree_cache[key]

def _compute_coverage_scores(unvisited_indices, points, current_node, destination_node, distance_matrix):
    """
    Compute spatial coverage penalty for each unvisited node.
    Returns array of penalty values for each unvisited node.
    """
    if len(unvisited_indices) == 0:
        return np.array([])
    
    unvisited_points = points[unvisited_indices]
    leaves = _get_quadtree(unvisited_points)
    
    # For each unvisited point, find which leaf it belongs to and compute area
    n_unvisited = len(unvisited_indices)
    area_penalty = np.zeros(n_unvisited)
    
    # Build mapping from point index in unvisited to leaf area
    # Each leaf contains a list of point indices (relative to unvisited set)
    for leaf in leaves:
        cx, cy, w, h, point_idxs = leaf
        area = w * h
        for pi in point_idxs:
            if pi < n_unvisited:
                area_penalty[pi] = area
    
    return area_penalty

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time, _LIMIT
    
    if (time.time() - _start_time) > _LIMIT:
        # Fallback: Nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        # Exact optimization for 2 nodes
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        
        if cost1 <= cost2:
            return c1
        else:
            return c2

    # General case: Quadtree-based spatial partitioning heuristic
    unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # Extract coordinates from distance matrix
    # Assuming Euclidean distance matrix, we can recover coordinates
    # But since we only have distance_matrix, we'll work with distances directly
    # We need to estimate spatial distribution from the distance matrix
    
    # Get distances from current node to all unvisited
    d_cur = distance_matrix[current_node, unvisited_arr]
    
    # Get distances from destination to all unvisited
    d_dest = distance_matrix[destination_node, unvisited_arr]
    
    # Compute a spatial coverage proxy using distance-based clustering
    # Nodes that are far from other unvisited nodes are in "sparse" regions
    # and should be prioritized to avoid leaving them for later
    
    # Compute pairwise distances among unvisited nodes
    if n_unvisited <= 50:
        # For small sets, compute full pairwise distances
        unvisited_set = set(unvisited_arr.tolist())
        pairwise_dists = np.zeros((n_unvisited, n_unvisited))
        for i in range(n_unvisited):
            for j in range(n_unvisited):
                pairwise_dists[i, j] = distance_matrix[unvisited_arr[i], unvisited_arr[j]]
        
        # For each node, compute its "isolation" - distance to nearest other unvisited node
        # and the area of its Voronoi-like region (approximated by average distance to k-nearest)
        k = min(3, n_unvisited - 1)
        isolation_scores = np.zeros(n_unvisited)
        
        for i in range(n_unvisited):
            # Get distances to all other unvisited nodes
            other_dists = np.delete(pairwise_dists[i], i)
            # Sort and take k-nearest
            sorted_dists = np.sort(other_dists)
            k_nearest = sorted_dists[:k]
            # Isolation score: average distance to k-nearest neighbors
            # Higher isolation = more "spatially expensive" to leave for later
            isolation_scores[i] = np.mean(k_nearest)
        
        # Normalize isolation scores
        if np.max(isolation_scores) > 0:
            isolation_scores = isolation_scores / np.max(isolation_scores)
        else:
            isolation_scores = np.ones(n_unvisited)
        
        # Composite score: balance distance to current, distance to destination, and isolation
        # We want to pick nodes that are:
        # 1. Close to current node (low d_cur)
        # 2. Not too far from destination (low d_dest) - to avoid long return trips
        # 3. In isolated regions (high isolation) - to prevent leaving them for later
        
        # Weighted combination
        w_dist = 1.0
        w_dest = 0.3
        w_isolation = 0.5
        
        # Normalize distances
        max_d_cur = np.max(d_cur) if np.max(d_cur) > 0 else 1.0
        max_d_dest = np.max(d_dest) if np.max(d_dest) > 0 else 1.0
        
        norm_d_cur = d_cur / max_d_cur
        norm_d_dest = d_dest / max_d_dest
        
        scores = w_dist * norm_d_cur + w_dest * norm_d_dest - w_isolation * isolation_scores
        
        # Pick the node with minimum score
        best_idx = np.argmin(scores)
        return int(unvisited_arr[best_idx])
    else:
        # For larger sets, use a simpler heuristic
        # Pick the node that minimizes d_cur + 0.3 * d_dest
        w_dist = 1.0
        w_dest = 0.3
        
        max_d_cur = np.max(d_cur) if np.max(d_cur) > 0 else 1.0
        max_d_dest = np.max(d_dest) if np.max(d_dest) > 0 else 1.0
        
        norm_d_cur = d_cur / max_d_cur
        norm_d_dest = d_dest / max_d_dest
        
        scores = w_dist * norm_d_cur + w_dest * norm_d_dest
        best_idx = np.argmin(scores)
        return int(unvisited_arr[best_idx])
