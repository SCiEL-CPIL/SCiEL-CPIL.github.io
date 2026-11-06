import time
import math

# EVOLVE-BLOCK-START
_start_time = None
_depot_pos_cache = None
_depot_pos_version = None

def _get_depot_position(depot, distance_matrix):
    """
    Estimate depot position by minimizing sum of squared distances to all nodes.
    Uses a simple gradient-free approach: the point that minimizes the sum of
    squared distances in 2D is the centroid, but we don't have coordinates.
    Instead, we use the depot's distances to define a "radial" coordinate system.
    """
    global _depot_pos_cache, _depot_pos_version
    # We'll just use the distance matrix row for depot as our reference
    return distance_matrix[depot]

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Explicit deadline guard
    # If we are close to the limit, fail safe by closing the route
    if time.time() - _start_time > 145.0:
        return depot

    # Identify feasible nodes
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if not feasible_nodes:
        return depot

    # Heuristic Selection
    if current_node == depot:
        # Start route: Nearest Neighbor to ensure we enter the "field" efficiently
        best_node = min(feasible_nodes, key=lambda n: distance_matrix[current_node, n])
    else:
        # Capacity-aware angular sweep:
        # Project nodes onto a circle centered at depot using their distance from depot
        # as radius, and angle derived from distances to current_node.
        # Select node with smallest angular deviation from current node's radial vector.
        
        d_curr_depot = distance_matrix[current_node, depot]
        
        if d_curr_depot < 1e-9:
            # Fallback to nearest neighbor if current is at depot
            best_node = min(feasible_nodes, key=lambda n: distance_matrix[current_node, n])
        else:
            # For each feasible node, compute a "score" based on angular alignment
            # Use the law of cosines to estimate the angle at depot between
            # (depot->current) and (depot->node)
            # cos(theta) = (d_depot_curr^2 + d_depot_node^2 - d_curr_node^2) / (2 * d_depot_curr * d_depot_node)
            
            best_node = feasible_nodes[0]
            best_score = -float('inf')
            best_dist = distance_matrix[current_node, best_node]
            
            for node in feasible_nodes:
                d_depot_node = distance_matrix[depot, node]
                d_curr_node = distance_matrix[current_node, node]
                
                if d_depot_node < 1e-9:
                    # Node is at depot (shouldn't happen for customers)
                    continue
                    
                # Compute cosine of angle at depot
                numerator = d_curr_depot * d_curr_depot + d_depot_node * d_depot_node - d_curr_node * d_curr_node
                denominator = 2.0 * d_curr_depot * d_depot_node
                
                if denominator < 1e-9:
                    continue
                    
                cos_theta = numerator / denominator
                # Clamp to [-1, 1] for numerical stability
                cos_theta = max(-1.0, min(1.0, cos_theta))
                
                # Angular alignment score: higher cos_theta means smaller angle (better alignment)
                # Also incorporate distance preference: closer nodes are better
                # Combine: score = cos_theta - alpha * (d_curr_node / max_d)
                # where alpha controls the tradeoff between alignment and distance
                
                # Use a simple combination: prefer high alignment (cos_theta close to 1)
                # and shorter distance
                score = cos_theta - 0.3 * (d_curr_node / max(d_curr_depot, 1.0))
                
                if score > best_score:
                    best_score = score
                    best_node = node
                    best_dist = d_curr_node
                elif score == best_score:
                    # Tie-break: choose closer node
                    if d_curr_node < best_dist:
                        best_node = node
                        best_dist = d_curr_node
                        
    return int(best_node)
# EVOLVE-BLOCK-END
