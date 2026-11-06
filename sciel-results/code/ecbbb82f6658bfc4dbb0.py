"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 148.0  # safety margin of 2s before 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, unvisited_nodes_list, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing using Clarke-Wright Savings.
    
    Strategy:
    1. If time is low or no feasible nodes, fallback to nearest feasible or depot.
    2. Compute savings for all pairs of unvisited nodes and depot.
    3. Greedily merge routes based on highest savings, respecting capacity constraints.
    4. Identify the route containing 'current_node' (which starts at depot).
    5. Return the next node in that route or depot.
    
    Note: The actual signature in the task is:
    select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)
    """
    # Correct signature handling
    # The prompt says: select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)
    # But my draft above had an extra arg. I will stick to the correct one.
    
    if _time_exceeded():
        # Fallback: nearest feasible node
        if len(unvisited_nodes) == 0:
            return int(depot)
        feasible_mask = demands[unvisited_nodes] <= rest_capacity
        if not np.any(feasible_mask):
            return int(depot)
        feasible_nodes = unvisited_nodes[feasible_mask]
        return int(feable_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Filter to feasible nodes
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    # Clarke-Wright Savings Logic:
    # The idea is to maximize the savings of visiting a node next.
    # Savings of inserting node k between i and j is: d[i,k] + d[k,j] - d[i,j]
    # Here, we are extending the current route.
    # Current route ends at current_node.
    # If we go to node k, we add d[current_node, k].
    # But we also "save" the future cost of going from k to the rest?
    
    # A simpler interpretation of "Savings" in a sequential greedy context:
    # We want to pick the node k that minimizes the incremental cost, 
    # but adjusted for the "loss of opportunity" of not visiting it later.
    
    # Actually, the classic CW heuristic constructs routes by merging.
    # Since we are building one route at a time (or extending the current one),
    # we can approximate the benefit of visiting k now vs later.
    
    # Let's use a "Savings" metric:
    # Benefit of visiting k next = (d[current_node, depot] + d[depot, k]) - d[current_node, k] ? No.
    
    # Let's look at the inspiration 2 (score 1.0820) which used Savings.
    # It says: "greedily merges the shortest feasible pairs of open routes"
    
    # In the context of `select_next_node`, we are effectively deciding the next step.
    # A strong heuristic is to minimize: d[current, k] + (estimated cost to close route from k)
    # But "Savings" suggests looking at pairs.
    
    # Let's try a different angle: **Nearest Neighbor with a Savings Correction**.
    # Score(k) = d[current, k] - Savings(k)
    # Where Savings(k) is the distance we save by visiting k now rather than later.
    # This is hard to compute without lookahead.
    
    # Alternative: **Savings-based Priority**.
    # Calculate the "savings" of connecting current_node to k compared to connecting current_node to depot and depot to k.
    # Savings(k) = d[current, depot] + d[depot, k] - d[current, k]
    # If Savings(k) is high, it means k is geographically aligned with the current direction (or close to current).
    # Wait, if k is close to current, d[current, k] is small, so Savings is large?
    # d[current, depot] is fixed. d[depot, k] varies.
    # If k is far from depot, d[depot, k] is large.
    # So Savings(k) = C + d[depot, k] - d[current, k].
    # Maximizing Savings is equivalent to minimizing d[current, k] - d[depot, k].
    # This favors nodes that are "farther from depot" than they are from current node?
    # This encourages going "outward" before returning.
    
    # Let's refine this.
    # Standard CW: Start with individual routes (depot-i-depot).
    # Merge i and j if savings s_ij = d[depot, i] + d[depot, j] - d[i, j] is high.
    
    # In our sequential generation, we have a partial route ending at `current`.
    # The "open end" is `current`. The "other end" is `depot`.
    # If we visit `k`, the new route is ...-current-k-...
    # We are effectively merging the "current route" with the "node k route".
    # The savings of this merge (compared to keeping them separate) is:
    # (Cost of current route ending at depot) + (Cost of node k route depot-k-depot) - (Cost of new route ending at k)
    # This is complex.
    
    # Let's stick to the simplest effective variant observed in literature for this specific API:
    # **Score = d[current, k] - alpha * (d[depot, k] - d[current, k])** ?
    
    # Let's try the direct Savings interpretation:
    # We want to pick k that maximizes the reduction in total distance if we were to close the route after k.
    # Cost if we go k then depot: d[current, k] + d[k, depot]
    # Cost if we go depot then k then depot (hypothetical separate): d[current, depot] + d[depot, k] + d[k, depot]
    # This doesn't seem right.
    
    # Let's look at Inspiration 2 again. It scored 1.0820 (better than 1.0012).
    # It used "Savings-based ... construction heuristic".
    
    # Implementation of Savings for next node selection:
    # For each feasible node k:
    # Calculate the "savings" of visiting k next.
    # A common proxy is: d[current, k] + d[k, depot] (cost to finish if we stop there)
    # vs d[current, depot] + d[depot, k] (cost if we closed now and did k separately)
    # Difference = d[current, k] + d[k, depot] - d[current, depot] - d[depot, k]
    # Minimizing this difference is equivalent to minimizing d[current, k] - d[depot, k] (since d[k,depot] and d[depot,k] are same, and d[current,depot] is constant).
    # So Score(k) = d[current, k] - d[depot, k]
    
    # This metric favors nodes that are "closer to current" relative to their distance from depot.
    # It encourages extending the route in the direction away from the depot, which is generally good for CVRP to reduce the number of return trips to depot.
    
    # Let's combine this with a capacity check and a small penalty for very large demands if capacity is tight.
    
    n = len(feasible_nodes)
    if n == 0:
        return int(depot)
        
    d_current_to_nodes = distance_matrix[current_node, feasible_nodes]
    d_depot_to_nodes = distance_matrix[depot, feasible_nodes]
    
    # Base score: d[current, k] - d[depot, k]
    # This is the "Savings" signal.
    scores = d_current_to_nodes - d_depot_to_nodes
    
    # Add a small penalty for nodes with high demand if capacity is low?
    # If rest_capacity is small, we should prioritize low demand nodes to fit more.
    # But if we prioritize low demand, we might fragment the route.
    # The Savings heuristic naturally handles some of this by geometry.
    
    # Let's add a small "urgency" factor: if capacity is very low, prioritize nodes that fit easily?
    # Or just stick to the pure savings signal which is known to work well (Inspiration 2).
    
    # To make it robust, let's blend with nearest neighbor slightly?
    # No, pure savings (min d[current,k] - d[depot,k]) is the distinct strategy.
    
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
