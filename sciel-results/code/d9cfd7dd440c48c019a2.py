"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard (optional, but good practice per directive)
# The task is small (50 nodes), so this is mostly for safety.
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # 10s margin before 120s limit


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next customer to visit using a dynamic regret-minimization heuristic.
    
    The score for each candidate is the negative of the difference between the current edge cost
    and the average edge cost of the remaining feasible set, effectively steering the route away
    from outlier-long edges to maintain uniform segment lengths and prevent premature capacity
    exhaustion on distant nodes.
    """
    # Check deadline (though N=50 is fast)
    if time.time() > DEADLINE:
        return 0

    # Convert unvisited_nodes to a numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes)
    
    if len(unvisited) == 0:
        return 0

    # Filter by capacity
    # demands is likely a list or array indexed by node id
    # We need to check if demands[node] <= rest_capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited])
    
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited[feasible_mask]

    # Calculate distances from current_node to all feasible nodes
    # distance_matrix is likely a 2D array
    d_current = distance_matrix[current_node, feasible_nodes]
    
    # Calculate distances from depot to all feasible nodes
    d_depot = distance_matrix[depot, feasible_nodes]

    # Base Clarke-Wright savings score
    base_scores = d_current - d_depot

    # Dynamic regret-minimization: penalize candidates with edge costs far above the mean
    # This steers the route away from outlier-long edges
    mean_edge_cost = np.mean(d_current)
    # The regret term is (d_current - mean_edge_cost), which is positive for long edges
    # We subtract this from the base score (which we want to minimize) to penalize long edges
    # Actually, we want to minimize base_scores, and we want to avoid long edges.
    # So we add a penalty for being above the mean.
    # Let's use a soft-min approach: we want to pick the node that minimizes the total cost.
    # The directive suggests: score = -(d_current - avg_edge_cost)
    # But we also have the Clarke-Wright term. Let's combine them.
    # If we just use the regret term, we might lose the directionality.
    # Let's try: scores = base_scores + lambda * (d_current - mean_edge_cost)
    # where lambda is a small positive value. This adds a penalty for long edges.
    
    # However, the directive says: "score for each candidate is the negative of the difference 
    # between the current edge cost and the average edge cost of the remaining feasible set"
    # This implies score = -(d_current - mean_edge_cost) = mean_edge_cost - d_current
    # Minimizing this score means maximizing (d_current - mean_edge_cost), i.e., picking the LONGEST edge?
    # That seems counter-intuitive for TSP. Let me re-read.
    # "effectively steering the route away from outlier-long edges"
    # If we pick the node that minimizes (d_current - mean), we pick the shortest edge relative to the mean.
    # If we pick the node that maximizes (d_current - mean), we pick the longest.
    # To steer AWAY from long edges, we should prefer short edges.
    # So we should minimize (d_current - mean_edge_cost).
    # But we also have the Clarke-Wright savings.
    # Let's try a weighted sum: scores = base_scores + alpha * (d_current - mean_edge_cost)
    # If alpha > 0, we penalize long edges.
    
    # Let's try alpha = 1.0 for simplicity, or maybe a smaller value like 0.5.
    # Let's also consider that base_scores can be negative (savings) or positive.
    # d_current and d_depot are positive.
    # base_scores = d_current - d_depot.
    # If d_current < d_depot, base_score is negative (good, we save distance).
    # The regret term (d_current - mean) is 0 on average.
    # So adding it will shift the scores but preserve the relative order mostly, unless d_current varies a lot.
    
    # Let's try a different interpretation:
    # The "regret" of picking a node is how much worse it is compared to the average.
    # We want to minimize regret. So we want to minimize (d_current - mean).
    # But we also want to maximize savings (minimize base_scores).
    # So: scores = base_scores + (d_current - mean_edge_cost)
    
    # Let's test this.
    scores = base_scores + (d_current - mean_edge_cost)
    
    # Find the index of the minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
