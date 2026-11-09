"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We'll implement a smarter heuristic that considers:
# 1. Feasibility (capacity constraint)
# 2. A combination of distance to candidate and distance from candidate back to depot
# 3. Demand consideration - prefer nodes that allow more future insertions

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot.
    
    Strategy: Among feasible nodes, pick the one that minimizes a combined cost:
    - Primary: distance from current to candidate
    - Secondary: distance from candidate to depot (penalize if going far from depot)
    - Consider demand: avoid taking a high-demand node if it would leave very little capacity
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Check if we can even fit any more nodes
    # If the minimum demand among unvisited exceeds rest_capacity, we must return to depot
    min_demand = min(demands[n] for n in unvisited_nodes)
    if min_demand > rest_capacity:
        return int(depot)
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return int(depot)
    
    # Score each feasible node
    # cost = dist(current, node) + alpha * dist(node, depot) + beta * demand_penalty
    # We want to minimize total route length, so:
    # - Prefer close nodes (saves travel now)
    # - Penalize nodes far from depot (if we go there, we'll have to travel back)
    # - Consider demand: taking a large demand early might be good or bad depending on context
    
    best_node = None
    best_score = float('inf')
    
    for n in feasible:
        d_to_node = distance_matrix[current_node, n]
        d_node_to_depot = distance_matrix[n, depot]
        demand = demands[n]
        
        # Combined heuristic score
        # The idea: the marginal cost of visiting n is d_to_node + d_node_to_depot 
        # (if this were the last node), but since we'll continue, it's more like
        # d_to_node + (expected future cost from n vs from current)
        # A simple approximation: weight the return-to-depot distance less when
        # there are many unvisited nodes left
        
        n_unvisited = len(unvisited_nodes)
        
        # If few nodes left, the return-to-depot distance matters more
        # If many nodes left, just go to the nearest feasible node
        if n_unvisited <= 3:
            # Near the end: strongly consider the return leg
            score = d_to_node + 0.8 * d_node_to_depot
        elif n_unvisited <= 8:
            score = d_to_node + 0.4 * d_node_to_depot
        else:
            score = d_to_node + 0.1 * d_node_to_depot
        
        # Small penalty for high demand when capacity is tight
        if rest_capacity < 10:
            score += 0.05 * demand
        
        if score < best_score:
            best_score = score
            best_node = n
    
    return int(best_node)
# EVOLVE-BLOCK-END
