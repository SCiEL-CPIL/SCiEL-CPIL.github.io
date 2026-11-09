"""Route-level look-ahead heuristic for Capacitated Vehicle Routing Problem.

Instead of scoring each candidate with a myopic combination of savings /
nearest-neighbor / demand-fit, we simulate the *completion* of the current
route under a fast nearest-neighbor-with-capacity policy for every feasible
candidate, and pick the candidate that minimizes the predicted total length
of the current route (already-traveled distance + predicted completion).
This is a different design family: a route-level surrogate objective with a
cheap constructive completion model, rather than a step-level greedy score.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT


def _complete_route_nn(start, remaining_cap, unvisited_set, demands, dist, depot):
    """
    Fast nearest-neighbor-with-capacity completion of a route.

    Returns (total_extra_distance, final_position) where total_extra_distance
    is the sum of leg distances from `start` through the greedily-chosen
    remaining nodes and back to the depot, and final_position is the depot
    (we always close at depot).

    This is intentionally cheap: O(k^2) with small k (remaining unvisited).
    """
    total = 0.0
    cur = start
    cap = remaining_cap
    remaining = list(unvisited_set)

    # Precompute demand array for fast lookup
    # We'll loop greedily; stop when no feasible node fits.
    while remaining:
        # Find feasible nodes
        # For speed, filter inline
        feasible = [n for n in remaining if demands[n] <= cap]
        if not feasible:
            break
        # Pick the nearest feasible node to cur
        best_n = None
        best_d = np.inf
        for n in feasible:
            d = dist[cur, n]
            if d < best_d:
                best_d = d
                best_n = n
        total += best_d
        cap -= demands[best_n]
        cur = best_n
        remaining.remove(best_n)
        if cap <= 0:
            break

    # Close at depot
    total += dist[cur, depot]
    return total, depot


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer by minimizing the predicted total length of the
    current route: simulate route completion under a fast NN-with-capacity
    policy for each feasible candidate, and choose the candidate whose
    simulated completion yields the shortest total route.

    Returns 0 (depot) if no feasible node or time limit reached.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = list(unvisited_nodes)
    dist = distance_matrix

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    # Distance already "spent" to reach current_node from depot is not
    # under our control for this decision; what matters is the *marginal*
    # total route length going forward. We compare:
    #   for candidate c: dist(current, c) + completion_from(c, cap - demand[c])
    # and pick the minimum.
    #
    # The unvisited set for completion excludes the chosen candidate.

    best_c = None
    best_score = np.inf

    # For speed on 50-node instances, the feasible set is small; this is fine.
    # We do a simple loop. If time is running out, fall back to nearest feasible.
    for c in feasible:
        if time.time() > _DEADLINE:
            break
        d_first = dist[current_node, c]
        new_cap = rest_capacity - demands[c]
        # Completion set: all feasible-except-c (and infeasible ones are
        # already excluded from the "remaining" pool for this route; but
        # infeasible-by-capacity nodes could still be visited later if cap
        # allows? No: if demands[n] > rest_capacity now, and we're only
        # *reducing* cap, they can never fit. So exclude them from completion.
        remaining_for_completion = [n for n in feasible if n != c]
        # Note: nodes infeasible now are never feasible later (cap only
        # decreases), so we can safely drop them.
        extra, _ = _complete_route_nn(c, new_cap, remaining_for_completion, demands, dist, depot)
        score = d_first + extra
        if score < best_score:
            best_score = score
            best_c = c

    if best_c is None:
        # Time ran out during the loop; fall back to nearest feasible node
        best_c = min(feasible, key=lambda n: dist[current_node, n])

    return int(best_c)
# EVOLVE-BLOCK-END
