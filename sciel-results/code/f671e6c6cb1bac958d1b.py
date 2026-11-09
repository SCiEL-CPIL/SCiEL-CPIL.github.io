"""TSP constructive heuristic: geometric convex hull peeling + slack blend."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 150.0 - 3.0  # seconds of wall-clock budget we actually use


def _now():
    return time.time()


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _now()
    return _DEADLINE - (_now() - _START_TIME)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction using a blend of:
      1. MST-leaf signal: nodes that are leaves in the MST of the remaining
         subgraph (unvisited + destination) are structurally "exposed" and
         should be visited next to avoid isolated subtours.
      2. Geometric slack: triangle-inequality slack as a proxy for turn angle.
      3. Direct distance: nearest-neighbor component.
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
    if _time_left() < 0.3:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])

    if n_unvisited == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a]
        return int(a) if cost_a <= cost_b else int(b)

    # --- Geometric Convex Hull Peeling ---
    # Compute the reference direction: from current_node toward destination_node
    # We need coordinates. Since we only have distance_matrix, we use a proxy:
    # For each candidate, compute the "turn angle" relative to the current->dest direction.
    #
    # Strategy: Use the triangle inequality slack as a proxy for how "outward" a node is.
    # A node that lies on or near the convex hull will have large slack when viewed
    # from the current->destination axis.
    
    d_cur_dest = distance_matrix[current_node, destination_node]
    d_cur_cand = distance_matrix[current_node, unvisited_nodes]
    d_dest_cand = distance_matrix[destination_node, unvisited_nodes]
    
    # Slack: how much "detour" each candidate adds relative to going straight to dest
    # Large slack = candidate is far from the direct path = likely on convex hull
    slack = d_cur_dest + d_cur_cand - d_dest_cand
    np.maximum(slack, 0.0, out=slack)
    
    # Normalize
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
    max_dist = np.max(d_cur_cand) if np.max(d_cur_cand) > 0 else 1.0
    
    slack_norm = slack / max_slack
    dist_norm = d_cur_cand / max_dist
    
    # --- 2-step lookahead: simulate greedy completion cost ---
    # For each candidate, estimate the cost of completing the tour greedily from there
    # This captures the actual trajectory better than static MST estimates
    
    # Precompute: for each candidate, what's the cheapest next node after it?
    # We do a lightweight 2-step lookahead:
    # cost_i = d(cur, cand_i) + min_j d(cand_i, cand_j) for j != i
    
    # To keep this fast, we limit the lookahead to a small subset
    # or use a vectorized approach
    
    # Vectorized 2-step lookahead:
    # For each candidate i, the best next step is min over j!=i of d(cand_i, cand_j)
    # Then total estimated cost = d(cur, i) + d(i, best_j) + d(best_j, dest)
    
    # Build candidate-to-candidate distance submatrix
    cand_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row, find the minimum (excluding self)
    # Set diagonal to inf to exclude self
    np.fill_diagonal(cand_dist, np.inf)
    min_next_dist = np.min(cand_dist, axis=1)  # min distance from each candidate to any other
    
    # Estimate 2-step cost: d(cur, cand) + min_next_dist + min distance from best_next to dest
    # Approximate: use the average or min of d(cand, dest) as proxy for the final leg
    
    # Simpler: 2-step cost = d(cur, cand) + min_next_dist
    # Then add a penalty for how far the best_next is from destination
    best_next_idx = np.argmin(cand_dist, axis=1)  # index of best next for each candidate
    best_next_to_dest = d_dest_cand[best_next_idx]
    
    lookahead_cost = d_cur_cand + min_next_dist + 0.3 * best_next_to_dest
    
    max_lookahead = np.max(lookahead_cost) if np.max(lookahead_cost) > 0 else 1.0
    lookahead_norm = lookahead_cost / max_lookahead
    
    # --- Blend scores ---
    # Combine: convex hull signal (slack), distance, and 2-step lookahead
    w_hull = 0.35    # weight for convex hull peeling (slack-based)
    w_dist = 0.25    # weight for direct distance
    w_look = 0.40    # weight for 2-step lookahead
    
    scores = w_hull * slack_norm + w_dist * dist_norm + w_look * lookahead_norm
    
    # Slight preference for keeping destination for closing the tour
    dest_mask = (unvisited_nodes == destination_node)
    if np.any(dest_mask) and n_unvisited > 8:
        scores[dest_mask] += 0.05 * dist_norm[dest_mask]
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
