import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145.0 # Safety margin

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Time guard
    if time.time() - _start_time > TIME_LIMIT:
        # Emergency exit: return nearest feasible or 0
        if not unvisited_nodes:
            return 0
        # Convert to list if needed for filtering
        un_list = list(unvisited_nodes)
        feasible = [u for u in un_list if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        # Pick nearest
        return int(min(feasible, key=lambda u: distance_matrix[current_node][u]))

    # Convert unvisited_nodes to numpy array for vectorized ops
    if not isinstance(unvisited_nodes, np.ndarray):
        unvisited_nodes = np.array(unvisited_nodes, dtype=int)
        
    if len(unvisited_nodes) == 0:
        return 0

    # Filter by capacity
    node_demands = demands[unvisited_nodes]
    feasible_mask = node_demands <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Get distances from current node to feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Get submatrix of distances between feasible nodes
    # This allows us to calculate the "exit cost" or "local density"
    # sub_matrix[i, j] = distance between feasible_nodes[i] and feasible_nodes[j]
    
    # For efficiency with small N (<=50), extracting the submatrix is fine.
    sub_matrix = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Calculate the minimum distance from each node to any OTHER feasible node.
    # This represents the "1-step lookahead" cost: if we go to u, where is the cheapest next step?
    # We exclude the node itself (diagonal is 0).
    
    # Set diagonal to infinity so argmin doesn't pick self
    np.fill_diagonal(sub_matrix, np.inf)
    
    min_dist_to_other = sub_matrix.min(axis=1)
    
    # If there's only 1 other node, min_dist_to_other is the distance to it.
    # If there are many, it's the nearest neighbor.
    
    # Heuristic Score:
    # Balance immediate cost (dist_from_current) and future cost (min_dist_to_other).
    # A weight of 0.5 is a standard choice for such heuristics.
    # We can also consider the depot distance if no other nodes are feasible, 
    # but min_dist_to_other handles the "next customer" aspect.
    # If we are the last customer, min_dist_to_other will be inf? 
    # No, if len(feasible_nodes) > 1, there is always at least one other.
    # If len(feasible_nodes) == 1, we returned early.
    
    # However, if the "other" node is not feasible? 
    # sub_matrix only contains feasible nodes. So min_dist_to_other is distance to another feasible node.
    # This is correct for 1-step lookahead within the feasible set.
    
    # What if we should go to depot? 
    # The function returns a node. If we pick a customer, we continue. 
    # If we want to close the route, we return 0.
    # The heuristic should implicitly decide when to close by picking a node that leads to a good route,
    # or we need a separate check. 
    # The seed code doesn't explicitly check "should I go to depot?". It just picks a customer.
    # The caller presumably stops when unvisited is empty or capacity is hit.
    # Wait, the contract says: "return 0 (depot) to close a route".
    # This implies the heuristic *can* choose to return 0.
    # When should it return 0?
    # Usually, if no feasible nodes, return 0.
    # Can it return 0 even if feasible nodes exist?
    # Yes, if the cost of going to the next customer is too high compared to returning to depot?
    # The seed code does NOT do this. It always picks a customer if one exists.
    # "Nearest feasible customer."
    # If we start returning 0 prematurely, we might leave feasible nodes unvisited, which is invalid?
    # "unvisited_nodes" implies these MUST be visited.
    # So we must visit all nodes in unvisited_nodes.
    # Therefore, we should only return 0 if unvisited_nodes is empty OR no feasible nodes.
    # The prompt says "picks the next customer, or the depot (0) to close a route".
    # If unvisited_nodes is not empty, we must pick a customer.
    # So, logic: if feasible_nodes is empty, return 0. Else pick best customer.
    
    scores = dist_from_current + 0.5 * min_dist_to_other
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
