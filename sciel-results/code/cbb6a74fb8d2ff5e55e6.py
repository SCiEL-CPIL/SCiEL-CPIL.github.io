"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_EVAL_START = time.time()
DEADLINE = _EVAL_START + 140  # Safety margin before 150s kill


def _time_remaining():
    return DEADLINE - time.time()


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a refined savings-inspired heuristic:
    - Filter to feasible customers (demand <= rest_capacity)
    - Score each candidate by a combination that approximates the savings
      from inserting the node into the current route vs. serving it separately.
    - Return depot if no feasible customer remains
    - Deadline guard to ensure we return before timeout
    """
    # Deadline guard
    if _time_remaining() < 1.0:
        # Emergency fallback: nearest feasible node
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # If no unvisited nodes, return depot
    if len(unvisited_nodes) == 0:
        return 0

    # Filter to only feasible customers (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible = unvisited_arr[feasible_mask]

    # If no feasible customer, return depot to close route
    if len(feasible) == 0:
        return 0

    # If only one feasible, return it
    if len(feasible) == 1:
        return int(feasible[0])

    # Savings-inspired scoring:
    # The savings from visiting node i next (instead of going back to depot
    # and then visiting i separately) is approximately:
    #   savings(i) = dist[current][depot] + dist[depot][i] - dist[current][i]
    # We want to maximize savings, i.e., minimize -savings(i).
    # But we also need to account for the remaining route cost.
    #
    # A refined approach: consider the marginal cost of adding node i to the
    # current route. The route so far is depot -> ... -> current.
    # If we go to i next, the additional cost is dist[current][i].
    # If we go to depot now and then start a new route to i, the cost is
    # dist[current][depot] + dist[depot][i].
    # The savings is the difference. We want to pick the node with maximum
    # savings (i.e., minimum marginal cost relative to closing the route).
    #
    # However, we should also consider the "future" cost: after visiting i,
    # how close is i to the depot? Nodes closer to the depot are "cheaper"
    # to close the route from, so we add a penalty for nodes far from depot.
    #
    # Score = dist[current][i] + alpha * dist[i][depot] - beta * (dist[current][depot] + dist[depot][i] - dist[current][i])
    # Simplifying: Score = (1+beta)*dist[current][i] + (alpha - beta)*dist[depot][i] - beta*dist[current][depot]
    # The last term is constant for all i, so:
    # Score = (1+beta)*dist[current][i] + (alpha - beta)*dist[depot][i]
    #
    # With alpha=0.3, beta=0.7:
    # Score = 1.7*dist[current][i] - 0.4*dist[depot][i]
    # This strongly favors nodes close to current and slightly penalizes
    # nodes far from depot (since -0.4 * dist[depot][i] means closer to depot
    # gives lower score).
    #
    # Actually, let me reconsider. The standard savings heuristic picks the
    # pair (i,j) that maximizes savings. In a sequential construction, we
    # approximate by picking the node that gives the best "savings" relative
    # to the current state.
    #
    # Let's use: score = dist[current][i] - 0.5 * (dist[current][depot] + dist[depot][i] - dist[current][i])
    #               = dist[current][i] - 0.5*dist[current][depot] - 0.5*dist[depot][i] + 0.5*dist[current][i]
    #               = 1.5*dist[current][i] - 0.5*dist[depot][i] - 0.5*dist[current][depot]
    # The last term is constant, so:
    # score = 1.5*dist[current][i] - 0.5*dist[depot][i]
    #
    # This means: strongly prefer nodes close to current, and mildly prefer
    # nodes close to depot (since -0.5*dist[depot][i] is lower when i is close).

    dist_to_node = distance_matrix[current_node, feasible]
    dist_to_depot = distance_matrix[feasible, depot]
    demands_feasible = demands[feasible]

    # Adaptive weights based on remaining capacity:
    # When capacity is tight, we should prefer customers that are:
    #   - Close to current node (to minimize travel)
    #   - Close to depot (to make it easier to close the route)
    #   - Low demand (to preserve capacity for more customers)
    
    # Capacity utilization: higher means tighter
    # Assume typical capacity is around 50 (from task description)
    # Use a normalized measure: rest_capacity / (rest_capacity + mean_demand)
    # When rest_capacity is small relative to demands, we're in tight capacity mode
    
    # Simple heuristic: weight of depot proximity increases as capacity gets tight
    # weight_depot = base + alpha * (1 - rest_capacity / (rest_capacity + 10))
    # When rest_capacity is large, weight_depot ≈ base
    # When rest_capacity is small, weight_depot ≈ base + alpha
    
    base_depot_weight = 0.5
    alpha = 0.3
    tightness = 1.0 - rest_capacity / (rest_capacity + 10.0)
    depot_weight = base_depot_weight + alpha * tightness
    
    # Demand penalty: when capacity is tight, penalize high-demand customers
    # to preserve capacity for more future customers
    demand_penalty_weight = 0.1 * tightness
    demand_penalty = demand_penalty_weight * demands_feasible / max(rest_capacity, 1)
    
    # Score: dist[current][i] - depot_weight * dist[depot][i] + demand_penalty
    # Lower score is better.
    scores = dist_to_node - depot_weight * dist_to_depot + demand_penalty

    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
