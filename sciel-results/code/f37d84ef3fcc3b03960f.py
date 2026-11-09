"""TSP constructive heuristic: geometric angle-minimization + 2-opt polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 150.0 - 5.0  # seconds of wall-clock budget we actually use


def _now():
    return time.time()


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _now()
    return _DEADLINE - (_now() - _START_TIME)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction.

    Structural change: instead of a 2-step min-distance lookahead, use a
    geometric turning heuristic. For each candidate c, compute the angle
    between the incoming edge (prev->current) and the outgoing edge
    (current->c); smaller angles keep the tour locally smooth (close to the
    convex hull / optimal TSP tour shape), which is a strong structural
    signal for Euclidean instances. Blend with a distance penalty so we don't
    chase tiny angles across the map.

    prev_node is recovered by scanning the distance_matrix column for the
    most recent visited node (the one not in unvisited and not the
    destination, with the smallest "visited-ness" heuristic: we can't know
    the full tour, so we use the destination as a proxy for the start and
    simply use the angle between (destination->current) and (current->c).
    This is a fixed, cheap geometric cue that is valid for the whole tour.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _now()

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Time guard: fall back to pure nearest-neighbour if we are close to the limit
    if _time_left() < 0.5:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])

    if n_unvisited == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        # Choose the one that yields the smaller total detour
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a]
        return int(a) if cost_a <= cost_b else int(b)

    # --- Geometric angle heuristic ---
    # Incoming direction: from destination (proxy for the start of the tour)
    # to the current node. Outgoing direction: from current to each candidate.
    # We use the distance_matrix to get 2D-ish geometry? No — distance_matrix
    # is only pairwise distances, not coordinates. So we CANNOT compute true
    # angles from the distance matrix alone.
    #
    # Therefore we fall back to a different structural idea that IS computable
    # from distances: the "detour" or "triangle-inequality slack". For each
    # candidate c, the slack of the path dest->current->c is
    #   slack(c) = d(dest,current) + d(current,c) - d(dest,c)
    # A small slack means c lies "between" dest and current, i.e. the turn is
    # gentle. This is a purely distance-based geometric proxy for angle.
    #
    # We also add the direct edge cost d(current,c) so we don't pick a far
    # node just because it has small slack.

    d_dc = distance_matrix[destination_node, current_node]
    d_cur_cand = distance_matrix[current_node, unvisited_nodes]          # (n,)
    d_dest_cand = distance_matrix[destination_node, unvisited_nodes]    # (n,)

    slack = d_dc + d_cur_cand - d_dest_cand
    np.maximum(slack, 0.0, out=slack)  # triangle inequality: slack >= 0

    # Blend: score = alpha * slack + (1-alpha) * d_cur_cand
    # alpha is a structural constant chosen to balance geometry vs. distance.
    alpha = 0.5
    scores = alpha * slack + (1.0 - alpha) * d_cur_cand

    # Slight preference for nodes that are not the destination until late,
    # to keep the destination for closing the tour (this is a structural
    # property of TSP tours: the destination is the start/end).
    dest_mask = (unvisited_nodes == destination_node)
    if np.any(dest_mask) and n_unvisited > 6:
        scores[dest_mask] += 0.01 * d_cur_cand[dest_mask]

    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
