# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time

    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        a, b = unvisited_nodes
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, destination_node]
        return int(a if cost_a <= cost_b else b)

    # Multi-criteria scoring: combine local distance, 2-step lookahead, and
    # a "convex-hull containment" proxy that penalizes candidates which push
    # the partial tour outward (away from the centroid of unvisited nodes).
    # This encourages boundary-expansion behaviour: pick nodes that keep the
    # partial tour compact and close to the remaining cluster.

    # Centroid of unvisited nodes (geometric containment proxy)
    # We don't have coordinates, so use the distance matrix to approximate:
    # a node that is "central" to the unvisited set has small average distance
    # to other unvisited nodes. We use this as a proxy for convex-hull containment.

    # Precompute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]

    # 2-step lookahead: for each candidate c, the best next step after c
    # (among remaining unvisited), plus cost to destination.
    # cost(c) = d(current, c) + min_{v in unvisited \ {c}} [d(c, v) + d(v, destination)]
    # But we also want a "containment" penalty: candidates far from the
    # unvisited centroid are penalised.

    # Approximate containment: average distance from candidate to other unvisited
    # (lower = more central = better containment)
    # This is O(n^2) to compute for all candidates at once.
    dist_matrix_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Set diagonal to 0 (self-distance)
    np.fill_diagonal(dist_matrix_unvisited, 0.0)
    avg_dist_to_unvisited = dist_matrix_unvisited.sum(axis=1) / max(n_unvisited - 1, 1)

    # 2-step lookahead cost
    # For each candidate i, the best next node j != i minimises d(unvisited[i], unvisited[j]) + d(unvisited[j], destination)
    # Precompute d(v, destination) for all unvisited
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]

    # For each i, find min over j != i of dist_matrix_unvisited[i, j] + dists_to_dest[j]
    # We can do this with a loop over i (n is small) or vectorize.
    # Vectorize: for each j, the value is dist_matrix_unvisited[:, j] + dists_to_dest[j]
    # Then for each i, we need the min over j != i.
    # Create a matrix where entry (i, j) = dist_matrix_unvisited[i, j] + dists_to_dest[j]
    lookahead_matrix = dist_matrix_unvisited + dists_to_dest[np.newaxis, :]
    # Set diagonal to inf so we don't pick j == i
    np.fill_diagonal(lookahead_matrix, np.inf)
    # For each i, the best next step cost is min over j of lookahead_matrix[i, j]
    best_next_cost = lookahead_matrix.min(axis=1)

    # Total score for candidate i:
    #   d(current, unvisited[i]) + best_next_cost[i] + alpha * avg_dist_to_unvisited[i]
    # The alpha term penalises candidates that are far from the rest of the
    # unvisited set (i.e., that would push the tour boundary outward).
    # We also add d(unvisited[i], destination) as a soft closing cost.
    alpha = 0.3  # weight for containment penalty
    scores = dists_from_current + best_next_cost + alpha * avg_dist_to_unvisited + 0.1 * dists_to_dest

    # Pick the candidate with the lowest score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
