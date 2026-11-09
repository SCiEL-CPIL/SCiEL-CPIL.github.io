"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = None

def _get_deadline():
    global _deadline
    if _deadline is None:
        _deadline = time.time() + 110.0
    return _deadline

def _time_exceeded():
    return time.time() > _get_deadline()

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Structural change: 2-step lookahead with capacity-urgency depot closure.
    
    Key insight: A 1-step nearest-neighbor + depot blend is a linear approximation.
    The true cost of choosing node i now is the 2-step path:
        dist(current, i) + min_j dist(i, j)   [j is the best next node after i]
    plus the eventual return to depot. When capacity is tight, we must close the
    route sooner, so we add a strong depot-return term that dominates as
    rest_capacity approaches the minimum feasible demand.
    
    Score for candidate i:
        d_cur[i] + w * min_j d_ij + gamma * d_depot[i]
    where w and gamma are capacity-adaptive.
    
    Additionally, if rest_capacity is so tight that only 1-2 nodes can fit,
    we strongly prefer nodes that are close to the depot (to close the route).
    """
    if len(unvisited_nodes) == 0:
        return depot

    if _time_exceeded():
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        # Pick the feasible node closest to depot to close route quickly
        return int(feasible[np.argmin(distance_matrix[feasible, depot])])

    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]

    if not feasible:
        return depot

    if len(feasible) == 1:
        return int(feasible[0])

    feasible_arr = np.array(feasible)
    unvisited_arr = np.array(unvisited_nodes)
    n_unvisited = len(unvisited_arr)

    # Distances from current to each feasible candidate
    d_cur = distance_matrix[current_node, feasible_arr]

    # Distances from each feasible candidate to depot
    d_depot = distance_matrix[feasible_arr, depot]

    # 2-step lookahead: for each candidate i, find the best next node j
    # (j can be any unvisited node, including infeasible ones for the purpose
    #  of measuring connectivity, but we prefer feasible j)
    # Shape: (len(feasible), n_unvisited)
    d_to_all = distance_matrix[feasible_arr[:, np.newaxis], unvisited_arr[np.newaxis, :]]

    # Mask out self
    same_node = (feasible_arr[:, np.newaxis] == unvisited_arr[np.newaxis, :])
    d_to_all_masked = np.where(same_node, np.inf, d_to_all)

    # For the 2-step lookahead, we want the min distance to ANY other unvisited node
    # (this represents the "continuation cost" of the route)
    min_next = np.min(d_to_all_masked, axis=1)
    min_next = np.where(np.isinf(min_next), 0.0, min_next)

    # Capacity urgency: estimate how many more nodes we can visit
    # Use the actual demands of unvisited nodes for a better estimate
    remaining_demands = demands[unvisited_arr]
    # Greedy estimate: sort by demand descending, count how many fit
    sorted_demands = np.sort(remaining_demands)[::-1]
    cumulative = np.cumsum(sorted_demands)
    max_nodes_fit = int(np.sum(cumulative <= rest_capacity))
    if max_nodes_fit == 0:
        max_nodes_fit = 1  # at least the current node fits

    # Urgency factor: 0 = no urgency (plenty of capacity), 1 = extreme urgency
    # Use a smooth sigmoid-like transition
    # When max_nodes_fit <= 1, we MUST close after this node -> high urgency
    # When max_nodes_fit >= 5, we have plenty of room -> low urgency
    if max_nodes_fit <= 1:
        urgency = 1.0
    elif max_nodes_fit <= 2:
        urgency = 0.7
    elif max_nodes_fit <= 3:
        urgency = 0.4
    elif max_nodes_fit <= 5:
        urgency = 0.15
    else:
        urgency = 0.05

    # When urgency is high, the depot return dominates the score
    # When urgency is low, the 2-step path cost dominates
    gamma = urgency * 1.5  # depot weight
    w = 1.0 - 0.5 * urgency  # 2-step lookahead weight (reduce when urgent)

    # Final score: lower is better
    scores = d_cur + w * min_next + gamma * d_depot

    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])

# EVOLVE-BLOCK-END
