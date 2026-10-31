"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 145.0  # seconds; safety margin before 150s kill
_START_TIME = time.time()


def _check_deadline():
    return time.time() > _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick next city using a structural change: 
    - For small remaining sets (<=8), use exact DP (Held-Karp) to find optimal completion
      and pick the first node in that optimal path.
    - For larger sets, use a multi-feature heuristic with adaptive weights:
      1. Nearest neighbor distance
      2. Destination bias (increases as unvisited shrinks)
      3. Regret/urgency: how much worse the 2nd-best option is
      4. 2-step lookahead for top candidates
    """
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited <= 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # Pick the one that leads to shorter total: d(curr,A)+d(A,B)+d(B,dest) vs d(curr,B)+d(B,A)+d(A,dest)
        a, b = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return a if cost_a <= cost_b else b

    # For small sets (<=8), use Held-Karp DP to find optimal tour completion
    if n_unvisited <= 8:
        return _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix)

    # For larger sets, use multi-feature heuristic
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[destination_node, unvisited_nodes]

    # Adaptive destination weight: stronger when few nodes remain
    if n_unvisited <= 10:
        w_dest = 0.5
        w_regret = 0.3
        w_lookahead = 0.2
    elif n_unvisited <= 20:
        w_dest = 0.3
        w_regret = 0.4
        w_lookahead = 0.3
    else:
        w_dest = 0.15
        w_regret = 0.45
        w_lookahead = 0.4

    # Normalize d_curr
    min_dc = np.min(d_curr)
    max_dc = np.max(d_curr)
    if max_dc > min_dc:
        d_curr_norm = (d_curr - min_dc) / (max_dc - min_dc)
    else:
        d_curr_norm = np.zeros_like(d_curr)

    # Normalize d_dest
    min_dd = np.min(d_dest)
    max_dd = np.max(d_dest)
    if max_dd > min_dd:
        d_dest_norm = (d_dest - min_dd) / (max_dd - min_dd)
    else:
        d_dest_norm = np.zeros_like(d_dest)

    # Regret: for each node, how much worse is the 2nd-best option vs this one
    # High regret means picking this node now is urgent
    sorted_d = np.sort(d_curr)
    if n_unvisited >= 2:
        second_best = sorted_d[1]
        regret = np.maximum(0.0, second_best - d_curr)
        min_r = np.min(regret)
        max_r = np.max(regret)
        if max_r > min_r:
            regret_norm = (regret - min_r) / (max_r - min_r)
        else:
            regret_norm = np.zeros_like(regret)
    else:
        regret_norm = np.zeros_like(d_curr)

    # 2-step lookahead: for each candidate, estimate cost of going to it then to its best next
    if n_unvisited <= 25:
        lookahead_costs = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            dists_from_i = distance_matrix[unvisited_nodes[i], unvisited_nodes].copy()
            dists_from_i[i] = np.inf
            min_next = np.min(dists_from_i)
            lookahead_costs[i] = d_curr[i] + 0.5 * min_next

        min_la = np.min(lookahead_costs)
        max_la = np.max(lookahead_costs)
        if max_la > min_la:
            la_norm = (lookahead_costs - min_la) / (max_la - min_la)
        else:
            la_norm = np.zeros_like(lookahead_costs)
    else:
        la_norm = np.zeros(n_unvisited)
        w_lookahead = 0.0

    # Combine scores (minimize)
    score = (1.0 - w_dest - w_regret - w_lookahead) * d_curr_norm + \
            w_dest * d_dest_norm + \
            w_regret * regret_norm + \
            w_lookahead * la_norm

    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])


def _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Use Held-Karp DP to find the optimal path from current_node through all unvisited
    nodes ending at destination_node. Return the first node in that optimal path.
    """
    n = len(unvisited_nodes)
    if n <= 1:
        return int(unvisited_nodes[0])
    
    # Map unvisited nodes to indices 0..n-1
    node_map = {int(unvisited_nodes[i]): i for i in range(n)}
    
    # dp[mask][i] = min cost to start at current_node, visit all nodes in mask,
    # and end at unvisited node i
    full_mask = (1 << n) - 1
    dp = np.full((1 << n, n), np.inf)
    parent = np.full((1 << n, n), -1, dtype=np.int32)
    
    # Initialize: from current_node to each unvisited node
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unvisited_nodes[i]]
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask, i] >= np.inf:
                continue
            # Try adding each node j not in mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask, i] + distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i
    
    # Find best ending node: dp[full_mask][i] + dist(unvisited[i], destination)
    best_end = -1
    best_cost = np.inf
    for i in range(n):
        total = dp[full_mask, i] + distance_matrix[unvisited_nodes[i], destination_node]
        if total < best_cost:
            best_cost = total
            best_end = i
    
    if best_end == -1:
        return int(unvisited_nodes[0])
    
    # Backtrack to find the first node
    mask = full_mask
    node = best_end
    first_node = node
    while parent[mask, node] != -1:
        node = parent[mask, node]
        mask = mask ^ (1 << first_node) if False else mask  # need to remove the node we're leaving
        # Actually, parent[mask][node] gives the previous node in the path
        # We need to reconstruct: the path ends at best_end, and we trace back
        # Let me redo the backtracking properly
        break
    
    # Proper backtracking
    mask = full_mask
    node = best_end
    path = []
    while node != -1:
        path.append(node)
        prev = parent[mask, node]
        if prev == -1:
            break
        # Remove current node from mask to get the mask before adding it
        mask = mask ^ (1 << node)
        node = prev
    
    # path is reversed: last element is the first node visited
    if path:
        first_idx = path[-1]
    else:
        first_idx = 0
    
    return int(unvisited_nodes[first_idx])
# EVOLVE-BLOCK-END
