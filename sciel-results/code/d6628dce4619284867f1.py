"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, fall back to nearest neighbor for safety
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Handle small cases with exact or near-exact logic
    if n_unvisited == 2:
        c1, c2 = unvisited_nodes[0], unvisited_nodes[1]
        # Cost if we pick c1: d(cur, c1) + d(c1, c2) + d(c2, dest)
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        # Cost if we pick c2: d(cur, c2) + d(c2, c1) + d(c1, dest)
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return int(c1 if cost1 <= cost2 else c2)
        
    if n_unvisited == 3:
        # Try all 3 permutations of picking the first node, then solving the rest optimally
        # Since 3 nodes remain, we can just evaluate the cost of each choice
        best_node = unvisited_nodes[0]
        best_cost = float('inf')
        for i, c in enumerate(unvisited_nodes):
            # If we pick c, the remaining 2 nodes must be visited in some order
            # Remaining: unvisited_nodes without c
            rem = np.delete(unvisited_nodes, i)
            r1, r2 = rem[0], rem[1]
            # Two possible orders for remaining: r1->r2 or r2->r1
            # Path: cur -> c -> r1 -> r2 -> dest
            cost_path1 = distance_matrix[current_node, c] + distance_matrix[c, r1] + distance_matrix[r1, r2] + distance_matrix[r2, destination_node]
            # Path: cur -> c -> r2 -> r1 -> dest
            cost_path2 = distance_matrix[current_node, c] + distance_matrix[c, r2] + distance_matrix[r2, r1] + distance_matrix[r1, destination_node]
            min_cost_for_c = min(cost_path1, cost_path2)
            if min_cost_for_c < best_cost:
                best_cost = min_cost_for_c
                best_node = c
        return int(best_node)

    # For n_unvisited >= 4, use 2-Opt Lookahead Heuristic
    # For each candidate c, find its closest neighbor n in the remaining set (excluding c).
    # Score c based on min(d(cur, c) + d(c, n), d(cur, n) + d(n, c))
    # This approximates the local cost of inserting c before n, considering the 2-opt swap.
    
    # Vectorized approach for speed
    # 1. Get distances from current to all unvisited
    dist_cur_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Get pairwise distances between unvisited nodes
    # sub_matrix is (N, N)
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. For each row i (candidate c_i), find the min distance to other nodes in the same row
    # We need to ignore the self-distance (diagonal)
    # Set diagonal to inf
    np.fill_diagonal(sub_matrix, np.inf)
    
    # min_dist_to_other[i] is the distance from unvisited[i] to its closest other unvisited node
    min_dist_to_other = np.min(sub_matrix, axis=1)
    
    # argmin_dist_to_other[i] is the index in unvisited_nodes of the closest neighbor
    argmin_idx = np.argmin(sub_matrix, axis=1)
    
    # The closest neighbor node for candidate i is unvisited_nodes[argmin_idx[i]]
    # We need d(current, closest_neighbor)
    # closest_neighbors = unvisited_nodes[argmin_idx]
    # dist_cur_to_closest = distance_matrix[current_node, closest_neighbors]
    
    # We can compute this directly
    dist_cur_to_closest = distance_matrix[current_node, unvisited_nodes[argmin_idx]]
    
    # Score calculation:
    # Option 1: Go to c, then to neighbor. Cost = d(cur, c) + d(c, neighbor)
    cost_direct = dist_cur_to_unvisited + min_dist_to_other
    
    # Option 2: "Swapped" logic. Cost = d(cur, neighbor) + d(neighbor, c)
    # Note: d(neighbor, c) is the same as d(c, neighbor) = min_dist_to_other
    # So cost_swapped = dist_cur_to_closest + min_dist_to_other
    
    # The heuristic prefers the path that looks locally cheaper.
    # If cost_swapped < cost_direct, it implies d(cur, neighbor) < d(cur, c), so c is further from current than its neighbor is.
    # By taking the min, we penalize candidates that are far from current relative to their local neighborhood.
    # Actually, we want to minimize the total tour length.
    # If we pick c, we *must* pay d(cur, c). The future cost is uncertain.
    # The lookahead estimates the cost of the next step.
    # If we pick c, the next step is likely to neighbor. So we pay d(c, neighbor).
    # Total estimated local cost: d(cur, c) + d(c, neighbor).
    # Why consider the swap? The swap d(cur, neighbor) + d(neighbor, c) is the cost if we had picked neighbor first.
    # If d(cur, neighbor) + d(neighbor, c) is significantly smaller than d(cur, c) + d(c, neighbor), it suggests that neighbor is a "better" next move than c.
    # However, we are forced to pick one.
    # A common heuristic is to score c by d(cur, c) + min_{n != c} (d(c, n) - d(cur, n))? No.
    
    # Let's stick to the simpler interpretation:
    # Score(c) = d(cur, c) + d(c, nearest_neighbor(c))
    # This is just a 2-step greedy.
    
    # To make it structural (2-Opt), we compare the direct path against the "cross" path.
    # If the cross path is cheaper, it indicates that c is "bad" because going to its neighbor first would have been cheaper.
    # We can penalize c if the cross path is much cheaper.
    # Score(c) = max(cost_direct, cost_swapped) ? No, that would avoid c if swap is cheap?
    # If swap is cheap, c is far. We should avoid c. So we want a HIGH score for c.
    # If direct is cheap, c is good. We want a LOW score for c.
    # So Score(c) = cost_direct + lambda * max(0, cost_swapped - cost_direct)
    # If cost_swapped > cost_direct, no penalty (direct is better).
    # If cost_swapped < cost_direct, penalty is (cost_swapped - cost_direct) which is negative? No.
    
    # Let's try: Score(c) = d(cur, c) + d(c, neighbor)
    # This is O(N^2) and simple. The 2-opt lookahead usually implies checking if the *insertion* of c into the *existing* tour can be improved. But here we are building the tour.
    
    # Alternative Structural Change: **Farthest Insertion** or **Cheapest Insertion** logic adapted for step-by-step.
    # But we can only pick one node.
    
    # Let's try a weighted combination of:
    # 1. Distance from current (greedy)
    # 2. Distance to destination (to avoid long tail)
    # 3. "Regret" of not picking the closest neighbor.
    
    # Let's use the "2-step lookahead" which is effectively what I calculated:
    # Score = d(cur, c) + d(c, best_next_from_c)
    
    # To make it stronger, let's look at the *second* best neighbor too?
    # Or let's use the 2-opt swap logic as a penalty.
    # If d(cur, n) + d(n, c) < d(cur, c) + d(c, n), then n is closer to cur than c is.
    # The "loss" of picking c over n is (d(cur, c) - d(cur, n)).
    # We want to minimize this loss.
    # So Score(c) = d(cur, c) + d(c, n) + max(0, d(cur, n) - d(cur, c))
    # If d(cur, n) > d(cur, c), no penalty.
    # If d(cur, n) < d(cur, c), penalty is d(cur, n) - d(cur, c).
    # This effectively makes Score(c) = d(cur, c) + d(c, n) + (d(cur, n) - d(cur, c)) = d(c, n) + d(cur, n) if n is closer.
    # This seems to favor c if c is close to cur, OR if c is very close to n (even if n is closer to cur? No).
    
    # Let's try a different structural approach: **Edge Weighted Nearest Neighbor**.
    # Instead of just distance, use the distance adjusted by the "density" of the unvisited set.
    
    # Given the stagnation, let's try **Or-Opt** style lookahead.
    # For each candidate c:
    # 1. Find the best place to insert c into the *remaining* tour? No, we are building it.
    
    # Let's go with the **2-Opt Lookahead** as defined:
    # Score(c) = min( d(cur, c) + d(c, n), d(cur, n) + d(n, c) )
    # This picks the candidate that results in the cheapest local 2-edge segment, assuming the next node is n.
    # This is a valid heuristic.
    
    scores = np.minimum(cost_direct, dist_cur_to_closest + min_dist_to_other)
    
    # Add a small weight to destination distance to prevent bad endings
    # dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    # scores += 0.1 * dist_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
