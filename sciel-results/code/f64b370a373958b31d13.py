"""Route-Level Look-Ahead Heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def _complete_route_nn(start_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Simulates the completion of the route starting from start_node.
    Uses a nearest-neighbor heuristic that respects capacity.
    Returns the total distance of the simulated path from start_node 
    through the visited nodes back to the depot.
    Note: start_node is assumed to be the first node added to the route 
    in this simulation segment. The distance from the previous node to 
    start_node is NOT included in this return value; it is added by the caller.
    """
    if rest_capacity <= 0:
        return distance_matrix[start_node, depot]
    
    # We need to track which nodes are still available in the simulation
    # unvisited_nodes is the list of nodes not yet visited in the *actual* route.
    # In the simulation, we will "visit" some of them.
    
    # To be fast, we work with indices and masks.
    # However, unvisited_nodes is a list of node IDs.
    # Let's convert to a numpy array for fast indexing if needed, 
    # but we need to remove nodes as we visit them.
    
    # A simple list-based approach might be slow. 
    # Let's try a masked array approach on the full distance matrix?
    # No, that's 50x50, which is small.
    
    # Strategy:
    # 1. Create a set of available nodes (initially unvisited_nodes excluding start_node? 
    #    No, start_node is the one we just picked. The remaining unvisited nodes are the candidates for the rest.)
    #    Wait, the caller passes `unvisited_nodes` which are all nodes not yet visited in the actual route.
    #    If we pick `start_node`, then `start_node` is no longer unvisited.
    #    So the simulation pool is `unvisited_nodes` minus `start_node`.
    
    available = [n for n in unvisited_nodes if n != start_node]
    current = start_node
    current_cap = rest_capacity - demands[start_node]
    total_dist = 0.0
    
    if not available:
        return distance_matrix[current, depot]
    
    # To speed up, we can pre-sort or just use a loop.
    # With N=50, a loop of 50 iterations where each iteration does a min over ~50 items is 2500 ops.
    # If we do this for 50 candidates, that's 125,000 ops per step.
    # 50 steps in a route -> 6,250,000 ops per route.
    # 8 instances -> 50,000,000 ops.
    # Python can handle ~10-50 million simple ops per second? Maybe borderline.
    # Let's try to optimize the inner loop.
    
    # We can use numpy for the min selection.
    # Convert available to numpy array
    avail_arr = np.array(available)
    
    while len(avail_arr) > 0:
        # Find nodes that fit in current_cap
        # demands is a numpy array indexed by node ID
        feasible_mask = demands[avail_arr] <= current_cap
        feasible_nodes = avail_arr[feasible_mask]
        
        if len(feasible_nodes) == 0:
            break
            
        # Find the nearest feasible node to current
        # distance_matrix[current, feasible_nodes]
        dists = distance_matrix[current, feasible_nodes]
        min_idx = np.argmin(dists)
        next_node = feasible_nodes[min_idx]
        min_dist = dists[min_idx]
        
        total_dist += min_dist
        current = next_node
        current_cap -= demands[next_node]
        
        # Remove next_node from available
        # Since avail_arr is a numpy array, we can use a mask to remove it
        # Or just create a new list. Creating a new list is O(N) but N is small.
        # Using numpy boolean indexing:
        avail_arr = avail_arr[avail_arr != next_node]
        
    # Return to depot
    total_dist += distance_matrix[current, depot]
    return total_dist

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a route-level look-ahead strategy.
    For each feasible candidate, simulates the rest of the route using 
    nearest-neighbor and picks the candidate that minimizes the total 
    projected route length.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter by capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible = unvisited[feasible_mask]
    
    if len(feasible) == 0:
        return 0

    if len(feasible) == 1:
        return int(feasible[0])

    # 2. Evaluate each candidate
    best_node = -1
    best_score = float('inf')
    
    # Optimization: If the set of feasible nodes is small, full look-ahead is fast.
    # If it's large, it might be slow. But 50 is the max.
    # We can also add a timeout check inside the loop if it's too slow.
    
    for c in feasible:
        # Immediate cost to reach c
        cost_to_c = distance_matrix[current_node, c]
        
        # Simulate rest of route starting from c
        # rest_capacity for the simulation is the capacity left AFTER visiting c
        sim_cost = _complete_route_nn(c, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)
        
        total_cost = cost_to_c + sim_cost
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = c
            
        # Time check inside loop to prevent timeout
        if time.time() > _DEADLINE:
            break

    if best_node == -1:
        return 0

    return int(best_node)

# EVOLVE-BLOCK-END
