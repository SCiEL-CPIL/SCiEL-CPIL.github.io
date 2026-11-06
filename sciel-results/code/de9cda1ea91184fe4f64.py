"""Rewritten TSP constructive heuristic with stronger algorithm."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

# Deadline guard
_DEADLINE = 140.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def _nearest_neighbor_tour(nodes, distance_matrix, start):
    """Build a tour using nearest neighbor from start through nodes."""
    if len(nodes) == 0:
        return 0.0
    remaining = set(nodes)
    current = start
    total = 0.0
    for _ in range(len(nodes)):
        if not remaining:
            break
        # Find nearest neighbor in remaining
        best_node = None
        best_dist = np.inf
        for node in remaining:
            d = distance_matrix[current, node]
            if d < best_dist:
                best_dist = d
                best_node = node
        total += best_dist
        current = best_node
        remaining.remove(current)
    return total


def _mst_lower_bound(nodes, distance_matrix):
    """Compute MST cost as a lower bound for the tour through nodes."""
    if len(nodes) <= 1:
        return 0.0
    
    n = len(nodes)
    node_list = list(nodes)
    
    # Prim's algorithm
    in_mst = [False] * n
    min_dist = [np.inf] * n
    min_dist[0] = 0
    total = 0.0
    
    for _ in range(n):
        # Find min dist node not in MST
        u = -1
        min_val = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < min_val:
                min_val = min_dist[i]
                u = i
        
        if u == -1:
            break
        
        in_mst[u] = True
        total += min_val
        
        # Update min_dist
        for v in range(n):
            if not in_mst[v]:
                d = distance_matrix[node_list[u], node_list[v]]
                if d < min_dist[v]:
                    min_dist[v] = d
    
    return total


def _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix, lookahead_depth=3):
    """
    Select next node using multi-step lookahead with MST lower bound.
    For each candidate, simulate going to it, then greedily picking nearest
    for lookahead_depth-1 more steps, then add MST lower bound for the rest.
    """
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    best_score = np.inf
    best_next = unvisited_nodes[0]
    
    for j_idx in range(n):
        j = unvisited_nodes[j_idx]
        
        # Cost to go from current to j
        cost = distance_matrix[current_node, j]
        
        # Simulate lookahead_depth-1 more greedy steps
        remaining = np.delete(unvisited_nodes, j_idx)
        current_pos = j
        
        for _ in range(lookahead_depth - 1):
            if len(remaining) == 0:
                break
            next_node = remaining[np.argmin(distance_matrix[current_pos, remaining])]
            cost += distance_matrix[current_pos, next_node]
            current_pos = next_node
            remaining = np.delete(remaining, np.where(remaining == next_node)[0][0])
        
        # Add MST lower bound for the rest (remaining + destination)
        if len(remaining) > 0:
            mst_nodes = np.append(remaining, destination_node)
            mst_lb = _mst_lower_bound(mst_nodes, distance_matrix)
            cost += 0.5 * mst_lb  # 0.5 * MST is a lower bound for TSP
        else:
            # No remaining nodes, just go to destination
            cost += distance_matrix[current_pos, destination_node]
        
        if cost < best_score:
            best_score = cost
            best_next = j
    
    return int(best_next)


def _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Use Held-Karp DP to find the optimal next node.
    For n <= 15, this is feasible.
    """
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    node_list = unvisited_nodes.astype(int)
    
    # dp[mask][j] = min cost to start at current_node, visit all nodes in mask,
    # and end at node_list[j]
    # Base: dp[1<<j][j] = dist(current_node, node_list[j])
    # Recurrence: dp[mask][j] = min over i in mask, i != j: dp[mask ^ (1<<j)][i] + dist(node_list[i], node_list[j])
    
    full_mask = (1 << n) - 1
    dp = np.full((1 << n, n), np.inf)
    
    for j in range(n):
        dp[1 << j, j] = distance_matrix[current_node, node_list[j]]
    
    for mask in range(1, 1 << n):
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            if mask == (1 << j):
                continue
            prev_mask = mask ^ (1 << j)
            for i in range(n):
                if prev_mask & (1 << i):
                    new_cost = dp[prev_mask, i] + distance_matrix[node_list[i], node_list[j]]
                    if new_cost < dp[mask, j]:
                        dp[mask, j] = new_cost
    
    # Find the next node: we want the first node in the optimal path
    # The total cost ending at j is dp[full_mask][j] + dist(node_list[j], destination_node)
    # But we need the FIRST node, not the last.
    
    # To find the first node, we can look at which j minimizes:
    # dist(current, j) + [optimal cost from j through all others to destination]
    
    # Let's compute for each candidate first node j:
    # cost_j = dist(current, j) + min over paths from j through all other unvisited to destination
    
    # This is equivalent to: for each j, compute the optimal path from j through
    # all other unvisited nodes to destination.
    
    best_cost = np.inf
    best_next = node_list[0]
    
    for j_idx in range(n):
        j = node_list[j_idx]
        remaining = np.delete(node_list, j_idx)
        
        if len(remaining) == 0:
            cost = distance_matrix[current_node, j] + distance_matrix[j, destination_node]
        else:
            # Compute optimal path from j through all remaining to destination
            m = len(remaining)
            # dp2[mask][k] = min cost to start at j, visit all nodes in mask, end at remaining[k]
            dp2 = np.full((1 << m, m), np.inf)
            for k in range(m):
                dp2[1 << k, k] = distance_matrix[j, remaining[k]]
            
            for mask in range(1, 1 << m):
                for k in range(m):
                    if not (mask & (1 << k)):
                        continue
                    if mask == (1 << k):
                        continue
                    prev_mask = mask ^ (1 << k)
                    for l in range(m):
                        if prev_mask & (1 << l):
                            new_cost = dp2[prev_mask, l] + distance_matrix[remaining[l], remaining[k]]
                            if new_cost < dp2[mask, k]:
                                dp2[mask, k] = new_cost
            
            full_mask2 = (1 << m) - 1
            # Find min over ending nodes
            min_suffix = np.min(dp2[full_mask2, :])
            # Find which ending node achieves this
            best_end_idx = np.argmin(dp2[full_mask2, :])
            cost = distance_matrix[current_node, j] + min_suffix + distance_matrix[remaining[best_end_idx], destination_node]
        
        if cost < best_cost:
            best_cost = cost
            best_next = j
    
    return int(best_next)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Picks the next city in a TSP tour.
    
    Strategy:
    - If there are few unvisited nodes (<= 15), use Held-Karp DP for exact optimal suffix.
    - Otherwise, use a lookahead heuristic with MST-based lower bound.
    - Time guard: fall back to nearest neighbor if running out of time.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, use simple nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small sets, use exact Held-Karp
    if n_unvisited <= 15:
        return _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # For larger sets, use lookahead heuristic with MST lower bound
    return _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix, lookahead_depth=3)
# EVOLVE-BLOCK-END
