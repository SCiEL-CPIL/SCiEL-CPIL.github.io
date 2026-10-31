"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

# Module-level deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 15.0  # seconds of safety margin before 150s limit
MAX_TIME_PER_INSTANCE = 10.0 # Max seconds to spend computing a tour for one instance

# Cache for pre-computed tours keyed by instance fingerprint
_tour_cache = {}

def _elapsed():
    return time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > 150.0 - DEADLINE_MARGIN

def _get_instance_fingerprint(distance_matrix):
    """
    Create a unique fingerprint for the distance matrix to cache tours.
    """
    flat_matrix = np.round(distance_matrix, decimals=4).flatten()
    return hash(flat_matrix.tobytes())

def _tour_length(tour, distance_matrix):
    """Calculate the total length of the closed tour."""
    n = len(tour)
    if n == 0:
        return 0
    next_tour = np.roll(tour, -1)
    return np.sum(distance_matrix[tour, next_tour])

def _estimate_remaining_cost(current_node, unvisited_nodes, destination_node, distance_matrix):
    """
    Heuristic estimate of the cost to complete the tour from current state.
    Uses:
    1. Distance from current to nearest unvisited (or destination if none left).
    2. Sum of minimum distances from each unvisited node to any other unvisited node (lower bound on internal tour).
    3. Distance from the "end" of the unvisited set back to destination (approximated by min dist to dest).
    
    This is a lower-bound-like heuristic to guide beam search.
    """
    if len(unvisited_nodes) == 0:
        return distance_matrix[current_node, destination_node]
    
    # Vectorized min distance for each unvisited node to any other unvisited node
    # This is O(N^2) per call, but we only do it for beam candidates.
    # To speed up, we can precompute min_out for all nodes once per instance? 
    # No, because unvisited set changes. 
    # However, for N=50, this is small.
    
    # 1. Cost to enter the unvisited set
    min_entry = np.min(distance_matrix[current_node, unvisited_nodes])
    
    # 2. Internal cost lower bound
    # For each node in unvisited, find min dist to any other node in unvisited
    # This approximates the cost of the path visiting them
    internal_costs = []
    unvisited_list = list(unvisited_nodes)
    for node in unvisited_list:
        # Min dist to other unvisited
        if len(unvisited_list) > 1:
            mask = np.ones(len(unvisited_list), dtype=bool)
            idx_self = unvisited_list.index(node)
            mask[idx_self] = False
            min_dist = np.min(distance_matrix[node, unvisited_list[mask]])
            internal_costs.append(min_dist)
        else:
            internal_costs.append(0)
    
    internal_cost = sum(internal_costs)
    
    # 3. Cost to exit to destination
    min_exit = np.min(distance_matrix[unvisited_nodes, destination_node])
    
    return min_entry + internal_cost + min_exit

def _beam_search_tour(distance_matrix, destination_node, max_time=MAX_TIME_PER_INSTANCE):
    """
    Construct a high-quality tour using Beam Search with heuristic lookahead.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    
    # Initial state: Start at 0 (or any node, but 0 is standard). 
    # Actually, TSP tour is cyclic. We can fix the start node to 0 to reduce symmetry.
    # The constructive process usually builds a path from a start to a destination.
    # The task implies building a tour step-by-step.
    # Let's assume the "tour" is built by adding nodes one by one.
    # The final step connects the last node to the start (or destination).
    # The function select_next_node takes current_node, destination_node, unvisited_nodes.
    # This suggests the "tour" is a path from some start to destination_node? 
    # Or is destination_node the start node of the original tour?
    # "destination_node" in TSP constructive context usually means the node we must return to, 
    # i.e., the start node.
    
    # Let's assume the tour starts at 0 and must return to 0.
    # So destination_node is likely 0.
    # But to be safe, we treat destination_node as the final return point.
    
    # Beam Search State: (heuristic_score, tour_so_far, current_node, unvisited_set)
    # We want to minimize the total tour length.
    # Total length = length_so_far + remaining_cost_estimate.
    
    # Initialize beam
    # Start node: 0
    start_node = 0
    initial_tour = [start_node]
    initial_unvisited = set(range(n)) - {start_node}
    initial_current = start_node
    
    # If destination is not 0, we might need to start elsewhere? 
    # Standard TSP: Start at 0, visit all, return to 0.
    # If destination_node != 0, it's a Path TSP? 
    # The prompt says "closed-tour length". So it's a cycle.
    # Usually, the "destination" in the loop is the start node.
    
    # Let's assume destination_node is the node we return to.
    # If the evaluator passes destination_node=0, we are good.
    
    beam = []
    # Score: length_so_far + heuristic
    # Initial length is 0. Heuristic estimates the rest.
    init_heur = _estimate_remaining_cost(initial_current, initial_unvisited, destination_node, distance_matrix)
    beam.append((init_heur, 0, initial_tour, initial_current, initial_unvisited))
    
    best_tour = None
    best_length = float('inf')
    
    beam_width = 5 # Keep top 5 paths
    max_depth = n
    
    for depth in range(1, max_depth + 1):
        if time.time() - start_time > max_time * 0.8:
            break
            
        new_beam = []
        # Expand all nodes in the current beam
        for score, _, tour, current, unvisited in beam:
            if not unvisited:
                # Tour complete
                # Add edge back to destination
                total_len = _tour_length(tour, distance_matrix) # This assumes tour is closed? 
                # _tour_length closes the loop. 
                # But our tour list is open [start, ..., last]. 
                # We need to add distance from last to destination.
                # Wait, _tour_length(tour, dm) computes sum dm[tour[i], tour[i+1]] with wrap.
                # If tour is [0, 1, 2], it computes 0->1, 1->2, 2->0.
                # So if we maintain an open path, we should just calculate the cost of the path + last->dest.
                
                # Let's calculate actual length properly for the final check.
                # Path length: sum dm[tour[i], tour[i+1]] for i=0..len-2
                # Plus dm[last, destination]
                path_len = 0
                for i in range(len(tour) - 1):
                    path_len += distance_matrix[tour[i], tour[i+1]]
                path_len += distance_matrix[tour[-1], destination_node]
                
                if path_len < best_length:
                    best_length = path_len
                    best_tour = tour[:]
                continue
            
            # Expand each unvisited node
            # To speed up, we can pre-filter candidates using the heuristic
            # For each candidate next_node in unvisited:
            # new_length = length_so_far + dm[current, next_node]
            # new_heur = _estimate_remaining_cost(next_node, unvisited - {next_node}, destination, dm)
            # total_est = new_length + new_heur
            
            # Optimization: We don't need to calculate full heuristic for every node if beam is small.
            # But N=50 is small.
            
            candidates = []
            for next_node in unvisited:
                edge_cost = distance_matrix[current, next_node]
                new_unvisited = unvisited - {next_node}
                # Heuristic for the rest
                # We can use a cheaper heuristic here: min dist from next_node to any unvisited + min dist from unvisited to dest
                # This is O(N) instead of O(N^2)
                if new_unvisited:
                    min_next_to_unvisited = np.min(distance_matrix[next_node, list(new_unvisited)])
                    min_unvisited_to_dest = np.min(distance_matrix[list(new_unvisited), destination_node])
                    fast_heur = min_next_to_unvisited + min_unvisited_to_dest
                else:
                    fast_heur = distance_matrix[next_node, destination_node]
                
                total_est = edge_cost + fast_heur
                candidates.append((total_est, next_node))
            
            # Sort candidates by estimated cost
            candidates.sort(key=lambda x: x[0])
            
            # Add top candidates to new_beam
            # We should add a few from each beam element to maintain diversity
            for est, next_node in candidates[:beam_width]:
                new_tour = tour + [next_node]
                new_current = next_node
                # Recalculate score for beam ordering
                # We use the same fast heuristic for consistency in the beam
                # Or we can store the cumulative length separately
                # Let's store (cumulative_length + heuristic, ...)
                # Actually, for beam search, we rank by total estimated final cost.
                # Cumulative length so far:
                cum_len = 0
                for i in range(len(new_tour) - 1):
                    cum_len += distance_matrix[new_tour[i], new_tour[i+1]]
                
                # For the beam, we just need to rank.
                # Using the fast heuristic is sufficient for ranking.
                new_beam.append((cum_len + fast_heur, cum_len, new_tour, new_current, new_unvisited))
        
        if not new_beam:
            break
            
        # Select top beam_width states from new_beam
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        # Check if we have a complete tour in the beam
        # (Already handled in the loop above if unvisited was empty)
        
    # If beam search didn't finish or found nothing, fallback to NN
    if best_tour is None:
        unvisited = set(range(n))
        tour = [0]
        unvisited.remove(0)
        current = 0
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            min_idx = np.argmin(dists)
            next_node = list(unvisited)[min_idx]
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        best_tour = tour
        best_length = _tour_length(tour, distance_matrix) # Closed tour length for logging, but we return the path
    
    return best_tour

def _get_tour(distance_matrix, destination_node):
    """
    Retrieve or compute the best tour for the given instance.
    """
    # Fingerprint should include destination if it affects the tour structure?
    # In a symmetric TSP, destination is usually 0. 
    # If destination changes, the "start" might effectively change or the path structure changes.
    # However, for a closed tour, the set of edges is the same regardless of where you "start" writing it down.
    # But the constructive process builds a path. 
    # If destination_node is always 0, we are fine.
    # Let's include destination_node in the fingerprint to be safe.
    fingerprint = (_get_instance_fingerprint(distance_matrix), destination_node)
    
    if fingerprint in _tour_cache:
        return _tour_cache[fingerprint]
    
    # Check deadline
    if _deadline_exceeded():
        # Fallback: simple NN tour
        n = distance_matrix.shape[0]
        unvisited = set(range(n))
        tour = [0]
        unvisited.remove(0)
        current = 0
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            min_idx = np.argmin(dists)
            next_node = list(unvisited)[min_idx]
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return tour
    
    # Compute tour
    tour = _beam_search_tour(distance_matrix, destination_node)
    
    # Cache it
    _tour_cache[fingerprint] = tour
    
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction.
    Uses Beam Search to determine the optimal sequence.
    """
    # Deadline guard
    if _deadline_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get or compute the pre-computed tour
    best_tour = _get_tour(distance_matrix, destination_node)
    
    # Find current node in the tour
    # The tour is a list [start, ..., last].
    # current_node should be in this list.
    try:
        current_idx = best_tour.index(current_node)
    except ValueError:
        # Fallback to NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # The next node in the tour
    if current_idx < len(best_tour) - 1:
        next_node_in_tour = best_tour[current_idx + 1]
    else:
        # We are at the end of the path. The next node should be the destination.
        # But destination_node might already be visited (it's the start).
        # In a constructive loop, if we are at the last node, the next "node" to pick is the destination to close the loop.
        # However, the function returns a node from unvisited_nodes.
        # If unvisited_nodes is empty, the loop shouldn't call this.
        # If unvisited_nodes is not empty, it means we haven't finished the path yet?
        # Wait, best_tour contains all nodes.
        # If current_node is the last node in best_tour, and unvisited_nodes is not empty,
        # it means our pre-computed tour doesn't match the current state?
        # This should not happen if we follow the tour strictly.
        
        # If we are at the last node of the pre-computed path, the next step is to go to destination_node.
        # But destination_node is likely the start node (0), which is already visited.
        # The function requires returning a node from unvisited_nodes.
        # If unvisited_nodes is empty, this function shouldn't be called.
        # If unvisited_nodes is NOT empty, it implies the pre-computed tour is invalid for this state.
        
        # Fallback: NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Check if this node is in the unvisited set
    if next_node_in_tour in unvisited_nodes:
        return int(next_node_in_tour)
    
    # Fallback: NN
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
# EVOLVE-BLOCK-END
