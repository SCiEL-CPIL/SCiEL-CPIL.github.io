"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a geometric convex-hull-peeling heuristic.
    
    The idea: instead of picking the nearest unvisited node, we try to follow
    the boundary of the unvisited point cloud. We compute the direction from
    current_node toward destination_node (or the centroid of unvisited nodes as
    a fallback), and then select the unvisited node that best continues this
    direction while staying close to the convex hull of the unvisited set.
    
    Specifically, we:
    1. Compute the vector from current_node to the centroid of unvisited nodes
       (as a proxy for "where we're going" in the tour).
    2. For each unvisited node, compute the angle between the current direction
       and the vector from current_node to that node.
    3. Also compute the distance from current_node to each unvisited node.
    4. Score each candidate by a combination of:
       - Angle deviation (prefer nodes that continue the current direction)
       - Distance (prefer nodes that are not too far, but not necessarily the closest)
       - A "hull-ness" factor: nodes on the convex hull of the unvisited set are
         preferred early in the tour.
    
    This encourages tracing the convex hull before filling interior points.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        # No unvisited nodes; return destination or current (shouldn't happen in valid TSP)
        return int(0)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get coordinates if available, otherwise work purely with distance matrix
    # Since we only have distance_matrix, we need to infer geometry or use
    # distance-based proxies for angle and hull-ness.
    
    # Direction vector proxy: use the vector from current to destination
    # But we don't have coordinates, only distances. So we need a different approach.
    
    # Alternative approach using only distance matrix:
    # 1. Compute distance from current to each unvisited node
    # 2. Compute distance from destination to each unvisited node  
    # 3. Use the "angle" proxy: nodes that are roughly equidistant from current
    #    and destination (or where the path current->node->destination is nearly
    #    straight) are preferred.
    
    # Actually, let's think differently. With only distance matrix, we can:
    # - Compute dist_current[i] = distance from current to unvisited[i]
    # - Compute dist_dest[i] = distance from destination to unvisited[i]
    # - Compute dist_curr_dest = distance from current to destination
    
    # A node that lies "between" current and destination on a straight line would
    # satisfy: dist_current[i] + dist_dest[i] ≈ dist_curr_dest
    # Nodes that deviate from this line have larger dist_current[i] + dist_dest[i]
    
    # For convex hull peeling, we want to follow the boundary. A good proxy:
    # Among unvisited nodes, prefer those that are "extreme" in some sense.
    
    # Let's use a combined score:
    # score[i] = alpha * dist_current[i] + beta * (dist_current[i] + dist_dest[i] - dist_curr_dest)
    # The second term penalizes nodes that are not on the "line" between current and dest.
    
    # But for convex hull tracing, we might want to prefer nodes that are FAR from
    # the centroid of unvisited nodes (i.e., on the hull). We can approximate this
    # by looking at how far each node is from all other unvisited nodes.
    
    # Simpler approach: use a "turning angle" heuristic.
    # We want to continue in the direction we're heading. Since we don't have
    # explicit coordinates, let's use the previous direction if available.
    
    # Actually, the function signature doesn't give us previous direction.
    # Let's use destination as the target direction.
    
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    dist_curr_dest = distance_matrix[current_node, destination_node]
    
    # Penalty for deviation from the straight line current->destination
    # Nodes on the line have dist_current + dist_dest ≈ dist_curr_dest
    # Nodes off the line have dist_current + dist_dest > dist_curr_dest
    line_penalty = dist_current + dist_dest - dist_curr_dest
    
    # "Hull-ness": nodes that are far from the centroid of unvisited nodes
    # Approximate centroid distance by average distance to other unvisited nodes
    if n_unvisited > 1:
        # Compute average distance from each unvisited node to all other unvisited nodes
        # This is expensive but n_unvisited is at most 50
        unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        avg_dist_to_others = unvisited_dist_matrix.sum(axis=1) / (n_unvisited - 1)
        # Nodes with high average distance are more likely on the hull
        hull_factor = avg_dist_to_others
    else:
        hull_factor = np.zeros(n_unvisited)
    
    # Normalize hull_factor to [0, 1]
    if hull_factor.max() > hull_factor.min():
        hull_factor_norm = (hull_factor - hull_factor.min()) / (hull_factor.max() - hull_factor.min())
    else:
        hull_factor_norm = np.zeros(n_unvisited)
    
    # Combine scores:
    # We want to minimize distance, but also prefer hull nodes and nodes on the line.
    # Let's use: score = dist_current * (1 + line_penalty / dist_curr_dest) - lambda * hull_factor_norm
    # The line_penalty term makes nodes off the line more expensive.
    # The hull_factor term makes hull nodes cheaper (preferred).
    
    if dist_curr_dest > 0:
        line_ratio = line_penalty / dist_curr_dest
    else:
        line_ratio = np.zeros(n_unvisited)
    
    # Clip line_ratio to avoid extreme values
    line_ratio = np.clip(line_ratio, 0, 2)
    
    # Score: lower is better
    # Base cost is distance, increased by line deviation, decreased by hull-ness
    score = dist_current * (1 + 0.5 * line_ratio) - 0.3 * hull_factor_norm * dist_current.mean()
    
    # Select the node with minimum score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
