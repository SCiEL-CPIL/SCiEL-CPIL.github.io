"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track route history for savings calculation
_route_start_time = None
_current_route = []  # List of nodes in current route (excluding depot)
_cache = {}

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _route_start_time, _current_route, _cache
    
    # Initialize deadline tracking
    if _route_start_time is None:
        _route_start_time = time.time()
    
    # Safety margin: stop if we've exceeded 140s (10s buffer before 150s limit)
    elapsed = time.time() - _route_start_time
    if elapsed > 140:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Check feasible nodes (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If this is the first node in the route (current_node is depot),
    # we don't have a "previous" node for savings calculation in the traditional sense.
    # We can fall back to nearest neighbor or use a different heuristic.
    # However, the savings heuristic is most powerful when extending a route.
    # For the first node, the "savings" compared to starting a new route is 0,
    # but we must pick one. Let's use a simple nearest neighbor for the first step
    # to establish a base, or better, use the savings logic if we track the route.
    
    # To properly implement Clarke-Wright savings, we need to know the previous node.
    # The interface `select_next_node` is stateless between calls unless we use globals.
    # We can reconstruct the "previous" node if we track the route.
    # But wait, `current_node` is provided. 
    # If `current_node` is depot (0), it's the first node.
    # If `current_node` is not depot, it's the last node added.
    
    # The savings heuristic typically compares:
    # Cost of separate routes: d(depot, A) + d(depot, B)
    # Cost of combined: d(depot, A) + d(A, B) + d(B, depot)
    # Savings = d(depot, A) + d(depot, B) - (d(depot, A) + d(A, B) + d(B, depot)) = d(depot, B) - d(A, B) ... wait.
    # Standard savings for linking A and B: s(A,B) = d(depot, A) + d(depot, B) - d(A, B).
    # When extending a route ending at `current_node` (A) with `candidate` (B),
    # we are effectively linking A and B.
    # The incremental cost of adding B to the route ending at A is d(A, B).
    # If we didn't add B to this route, we'd have to start a new route or add it elsewhere.
    # A common approximation for "insertion" heuristics is to minimize the incremental cost:
    # d(prev, current) is already paid.
    # Cost to go from current to candidate and then eventually to depot?
    
    # Let's use a simpler, effective heuristic: **Cheapest Insertion** or **Savings-based**.
    # Since we are building a route sequentially (current -> next), we can score each candidate `c` by:
    # Score = d(current, c) + d(c, depot) - d(current, depot)
    # This is the "savings" of going current->c->depot instead of current->depot (if we were closing) 
    # BUT this doesn't account for future nodes.
    
    # A better metric for sequential construction is the **Savings Heuristic** adapted for insertion:
    # We want to pick `c` such that it creates the most "value".
    # Let's look at the change in total distance if we insert `c` between `current` and `depot` (assuming we close after `c`).
    # Current route cost so far: ... -> current.
    # If we close now: cost += d(current, depot).
    # If we add c then close: cost += d(current, c) + d(c, depot).
    # Difference: d(current, c) + d(c, depot) - d(current, depot).
    # We want to minimize this difference? No, we want to maximize the "savings" relative to NOT adding it.
    # Actually, this metric favors nodes that are "in the way" of the return to depot.
    
    # Let's try a different approach: **Savings between current and candidate**.
    # If we are at `current` and consider `candidate`, the cost to go to candidate is d(current, candidate).
    # If we were to go to depot, it's d(current, depot).
    # If we go to candidate, we can potentially continue.
    
    # Let's use the **Savings** defined as:
    # s = d(depot, current) + d(depot, candidate) - d(current, candidate)
    # This is the standard Clarke-Wright savings for linking current and candidate.
    # Higher savings is better.
    # However, we must also consider capacity.
    
    # Let's calculate savings for all feasible nodes.
    # d_depot_current = distance_matrix[depot, current_node]
    # d_depot_cand = distance_matrix[depot, feasible_nodes]
    # d_current_cand = distance_matrix[current_node, feasible_nodes]
    # savings = d_depot_current + d_depot_cand - d_current_cand
    
    # We want to MAXIMIZE savings.
    
    d_depot_current = distance_matrix[depot, current_node]
    d_depot_cand = distance_matrix[depot, feasible_nodes]
    d_current_cand = distance_matrix[current_node, feasible_nodes]
    
    savings = d_depot_current + d_depot_cand - d_current_cand
    
    # We also need to ensure that the path is efficient.
    # Sometimes high savings might lead to a very long detour if d_current_cand is small but d_depot_cand is large?
    # No, the formula handles that.
    
    # Pick the node with max savings.
    # Tie-breaker: if savings are equal, pick the one with lower distance from current?
    # Or just argmax.
    
    idx = np.argmax(savings)
    best_node = feasible_nodes[idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
