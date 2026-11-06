"""Constructive TSP heuristic: 2-step lookahead nearest neighbour.

MODEL-AND-OPTIMIZE:
    We model the cost contribution of picking candidate c (from current node)
    as the immediate edge cost plus a lookahead estimate of the next edge:
        score(c) = dist(current, c) + beta * next_cost(c)
    where next_cost(c) is:
      - min distance from c to any other unvisited node (if |unvisited| > 1)
      - dist(c, destination) (if c is the last unvisited node)
    This 2-step lookahead avoids locally optimal but globally suboptimal
    choices by penalizing candidates that lead to expensive subsequent edges.
    The weight beta is adapted: higher when many nodes remain (lookahead is
    more informative) and lower near the end (pure NN is more reliable).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Explicit deadline guard: the evaluator kills after 150 s wall-clock.
# Each call is O(|unvisited|) vectorised NumPy work (microseconds for n=50),
# so this guard is only a safety net; it does not gate the search.
_RUN_START = time.perf_counter()
_DEADLINE_S = 110.0  # safety margin well under the 150 s kill switch


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Safety: if (unexpectedly) no candidates remain, close to the destination.
    if unvisited_nodes is None or len(unvisited_nodes) == 0:
        return int(destination_node)

    # Deadline guard: if we are dangerously close to the wall-clock limit,
    # fall back to the cheapest deterministic choice (pure NN) immediately.
    if (time.perf_counter() - _RUN_START) > _DEADLINE_S:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    # Adaptive beta: start moderate (0.5) to use lookahead, decay to low (0.1)
    # near the end where lookahead is less useful.
    fraction_remaining = n_unvisited / 49.0
    beta = 0.1 + 0.4 * fraction_remaining

    # Immediate edge costs from current node to each candidate.
    d_current = distance_matrix[current_node, unvisited_nodes]

    # Lookahead cost: for each candidate c, estimate the cost of the next edge.
    if n_unvisited == 1:
        # Last candidate: the next edge goes to destination.
        next_cost = distance_matrix[unvisited_nodes[0], destination_node]
        scores = d_current + beta * next_cost
    else:
        # For each candidate, find the minimum distance to any OTHER unvisited node.
        # Use a matrix of distances between all pairs of unvisited nodes.
        # dist_matrix[unvisited_nodes, :][:, unvisited_nodes] gives pairwise distances.
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Set diagonal to inf so we don't pick the node itself as the "next" node.
        np.fill_diagonal(sub_dist, np.inf)
        # For each candidate i, the lookahead is min_j dist(i, j) for j != i.
        # But we also need to consider the destination if it's the last step.
        # For simplicity, use min distance to other unvisited nodes.
        min_next = np.min(sub_dist, axis=1)
        scores = d_current + beta * min_next

    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
