"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 3-step lookahead heuristic.
    For each candidate i, we look ahead to the best next node j (j != i),
    and then estimate the cost to reach the destination from j.
    
    Score(i) = dist(current, i) + min_{j in U, j != i} [dist(i, j) + dist(j, destination)]
    
    This 3-step lookahead (current -> i -> j -> destination) provides a better
    estimate of the future cost than the 2-step lookahead, helping to avoid
    choices that lead to expensive final connections.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # Only two choices: pick the one that leads to the shorter total path
        u0, u1 = unvisited_nodes
        cost0 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        if cost0 <= cost1:
            return int(u0)
        else:
            return int(u1)
    
    U = unvisited_nodes
    d_cU = distance_matrix[current_node, U]
    d_UD = distance_matrix[U, destination_node]
    d_UU = distance_matrix[np.ix_(U, U)]
    
    # cost_next[i, j] = dist(U[i], U[j]) + dist(U[j], destination)
    cost_next = d_UU + d_UD[np.newaxis, :]
    
    # Exclude j = i by setting diagonal to infinity
    np.fill_diagonal(cost_next, np.inf)
    
    # For each i, find the minimum cost to go from U[i] to some j != i and then to destination
    min_next_cost = np.min(cost_next, axis=1)
    
    # Total score for choosing U[i] as the next node
    scores = d_cU + min_next_cost
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])
# EVOLVE-BLOCK-END
