"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 10.0  # seconds before 150s limit


def _check_deadline():
    """Return True if deadline is approaching."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using a hybrid heuristic combining:
    1. Nearest neighbor distance from current node
    2. Lookahead to destination (to avoid long closing edge)
    3. MST-leaf preference: prefer nodes that are leaves in the MST of unvisited+destination,
       as these represent the global skeleton of the tour
    
    The score is a weighted combination that balances local proximity with global structure.
    """
    if _check_deadline():
        # Safety fallback: nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from unvisited nodes to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute MST of the subgraph induced by unvisited_nodes + destination_node
    # For efficiency with up to ~50 nodes, we can use a simple Prim's algorithm
    mst_nodes = np.append(unvisited_nodes, destination_node)
    n_mst = len(mst_nodes)
    
    # Build local distance matrix for MST nodes
    mst_dist = distance_matrix[np.ix_(mst_nodes, mst_nodes)]
    
    # Prim's algorithm to find MST and identify leaves
    in_mst = np.zeros(n_mst, dtype=bool)
    min_edge = np.full(n_mst, np.inf)
    min_edge[0] = 0.0
    parent = np.full(n_mst, -1, dtype=int)
    
    for _ in range(n_mst):
        # Find unvisited node with minimum edge
        unvisited_mask = ~in_mst
        if not np.any(unvisited_mask):
            break
        candidates = np.where(unvisited_mask)[0]
        u = candidates[np.argmin(min_edge[candidates])]
        in_mst[u] = True
        
        # Update min_edge for neighbors
        for v in range(n_mst):
            if not in_mst[v] and mst_dist[u, v] < min_edge[v]:
                min_edge[v] = mst_dist[u, v]
                parent[v] = u
    
    # Count degree in MST to identify leaves
    degree = np.zeros(n_mst, dtype=int)
    for v in range(1, n_mst):
        if parent[v] >= 0:
            degree[v] += 1
            degree[parent[v]] += 1
    
    # Map back to unvisited_nodes indices
    # mst_nodes[0:n_unvisited] are unvisited_nodes, mst_nodes[n_unvisited] is destination
    is_leaf = np.zeros(n_unvisited, dtype=bool)
    for i in range(n_unvisited):
        if degree[i] <= 1:
            is_leaf[i] = True
    
    # Combine scores:
    # - Primary: distance from current node (nearest neighbor)
    # - Secondary: distance to destination (lookahead)
    # - Tertiary: MST leaf preference (global structure)
    
    # Normalize distances for blending
    max_dist_current = np.max(dist_from_current) if n_unvisited > 0 else 1.0
    max_dist_dest = np.max(dist_to_dest) if n_unvisited > 0 else 1.0
    
    # If max is 0, avoid division by zero
    if max_dist_current == 0:
        max_dist_current = 1.0
    if max_dist_dest == 0:
        max_dist_dest = 1.0
    
    norm_dist_current = dist_from_current / max_dist_current
    norm_dist_dest = dist_to_dest / max_dist_dest
    
    # Weighted combination:
    # 60% nearest neighbor, 25% lookahead to destination, 15% MST leaf bonus
    # MST leaf bonus: leaves get 0 (good), non-leaves get 1 (penalty)
    mst_penalty = np.where(is_leaf, 0.0, 1.0)
    
    score = 0.60 * norm_dist_current + 0.25 * norm_dist_dest + 0.15 * mst_penalty
    
    # Select node with minimum score
    best_idx = np.argmin(score)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
