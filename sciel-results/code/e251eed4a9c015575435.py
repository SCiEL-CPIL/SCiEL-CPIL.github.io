"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

# Module-level state to track time and cache computations
_start_time = None
_time_limit = 120.0  # seconds
_safety_margin = 5.0  # seconds
_cache = {}
_held_karp_threshold = 12  # Use Held-Karp DP for last N nodes


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_exceeded():
    return _get_elapsed() > (_time_limit - _safety_margin)


def _held_karp_dp(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Solve the TSP suffix exactly using Held-Karp DP for up to _held_karp_threshold nodes.
    Returns the best next node to visit.
    """
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Map unvisited nodes to indices 0..n-1
    node_map = {node: i for i, node in enumerate(unvisited_nodes)}
    
    # dp[mask][i] = minimum cost to visit all nodes in mask (starting from i) and end at i
    # We want to find the minimum cost path starting from current_node, visiting all nodes in unvisited_nodes,
    # and returning to destination_node.
    
    # State: (mask, last_node) where mask is a bitmask of visited nodes among unvisited_nodes
    # dp[mask][last] = min cost to have visited exactly the set 'mask' and be at 'last'
    
    # Initialize: dp[0][i] = dist(current_node, unvisited_nodes[i]) for all i
    # Then for each state, extend to unvisited nodes.
    
    # Number of states: 2^n * n
    # For n=12, that's 4096 * 12 = 49152 states, each with up to 12 transitions -> ~590k ops. Fast.
    
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    
    # Initialize: start from current_node, go to each unvisited node
    for i in range(n):
        dp[0][i] = distance_matrix[current_node, unvisited_nodes[i]]
    
    # Fill DP table
    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                continue  # last must be in mask
            # Try to extend to each unvisited node not in mask
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue  # already visited
                new_mask = mask | (1 << nxt)
                cost = dp[mask][last] + distance_matrix[unvisited_nodes[last], unvisited_nodes[nxt]]
                if cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = cost
    
    # Find the best completion: min over all last nodes of dp[full_mask][last] + dist(last, destination)
    full_mask = (1 << n) - 1
    best_cost = INF
    best_last = -1
    for last in range(n):
        total = dp[full_mask][last] + distance_matrix[unvisited_nodes[last], destination_node]
        if total < best_cost:
            best_cost = total
            best_last = last
    
    # Now we need to find the first node to visit. We can backtrack or recompute.
    # Actually, we need to find which node to visit first from current_node.
    # The DP gives us the optimal path, but we need the first step.
    # Let's recompute the first step by checking which first node leads to the optimal total.
    
    # Alternative: reconstruct the path by backtracking from the best_last.
    # But we just need the first node. Let's compute it directly.
    
    # For each candidate first node i, the cost would be:
    # dist(current, i) + (optimal cost to visit remaining nodes from i and return to dest)
    # This is exactly what the DP computes, but we need to extract it.
    
    # Actually, the DP as formulated doesn't directly give us the cost starting from a specific first node.
    # Let's restructure: we want min over first node i of [dist(current, i) + optimal_suffix(i, remaining, dest)]
    
    # Let's use a different DP formulation:
    # f(mask, last) = min cost to visit all nodes in mask (where mask includes last) and end at last,
    # then return to destination.
    # f(mask, last) = min over nxt not in mask of [dist(last, nxt) + f(mask | {nxt}, nxt)]
    # Base case: f({last}, last) = dist(last, destination)
    
    # This is cleaner. Let's redo.
    
    dp2 = [[INF] * n for _ in range(1 << n)]
    
    # Base case: mask with only last
    for i in range(n):
        dp2[1 << i][i] = distance_matrix[unvisited_nodes[i], destination_node]
    
    # Fill in reverse order of mask size
    for mask in range((1 << n) - 1, 0, -1):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if mask == (1 << last):
                continue  # already set
            # Try to extend from last to any nxt not in mask
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                cost = distance_matrix[unvisited_nodes[last], unvisited_nodes[nxt]] + dp2[new_mask][nxt]
                if cost < dp2[mask][last]:
                    dp2[mask][last] = cost
    
    # Now, the first node i has total cost: dist(current, i) + dp2[1 << i][i]
    # Wait, dp2[1 << i][i] is the cost from i to visit no other nodes and return to dest.
    # But we need to visit ALL nodes. So we need dp2 for the full set starting from i.
    # Actually, dp2[mask][last] is defined as the cost to visit all nodes in mask (starting from last, with last in mask)
    # and return to destination. But this assumes we start at last and must visit all nodes in mask.
    # So for the first node i, the total cost is:
    # dist(current, i) + (cost to visit all nodes in unvisited_nodes starting from i and returning to dest)
    # But dp2 as defined doesn't capture "starting from i and visiting all nodes in unvisited_nodes".
    
    # Let me redefine clearly:
    # Let g(mask, last) = minimum cost to start at 'last', visit all nodes in 'mask' (where 'last' is in 'mask'),
    # and return to destination_node.
    # g({last}, last) = dist(last, dest)
    # g(mask, last) = min over nxt in mask\{last} of [dist(last, nxt) + g(mask\{last}, nxt)]
    # Wait, that's not right either. If we're at 'last' and need to visit all nodes in 'mask', we can go to any nxt in mask\{last},
    # then from nxt we need to visit all nodes in mask\{last} (including nxt? No, nxt is already visited).
    
    # Correct formulation:
    # g(mask, last) = min cost to have just arrived at 'last', having visited exactly the set 'mask' (last in mask),
    # and then visit all remaining nodes (unvisited_nodes \ mask) and return to destination.
    # Base case: if mask == full_set, g(full_set, last) = dist(last, dest)
    # Otherwise: g(mask, last) = min over nxt not in mask of [dist(last, nxt) + g(mask | {nxt}, nxt)]
    
    # This is what my first DP did. Let's go back to that and just extract the first node.
    
    # Using the first DP formulation:
    # dp[mask][last] = min cost to have visited exactly 'mask' and be at 'last'
    # dp[0][i] = dist(current, i)
    # dp[mask | {nxt}][nxt] = min(dp[mask][last] + dist(last, nxt)) for all last in mask
    # Answer: min over last of dp[full_mask][last] + dist(last, dest)
    
    # To find the first node, we check for each i:
    # total_cost_if_first_i = dist(current, i) + (min cost to visit all other nodes starting from i and return to dest)
    # The second term is: min over paths from i visiting all other nodes and returning to dest.
    # This is equivalent to: let mask_rest = full_mask \ {i}, then we need:
    # min over last in mask_rest of [dp_rest[mask_rest][last] + dist(last, dest)]
    # where dp_rest is computed starting from i.
    
    # This is getting complicated. Let's just use a simpler approach:
    # For each candidate first node i, compute the total tour length if we choose i first,
    # then solve the remaining TSP exactly (with n-1 nodes).
    # For n=12, that's 12 subproblems of size 11, each with 2^11 * 11 = 22528 states. Total ~270k ops. Fine.
    
    # But we already computed the full DP. Let's just backtrack.
    
    # Actually, the simplest way: we know the optimal total cost. For each first node i,
    # the cost is dist(current, i) + h(i) where h(i) is the optimal cost to visit all remaining nodes
    # starting from i and returning to dest.
    
    # Let's compute h(i) using the DP.
    # h(i) = min over paths from i visiting all nodes in unvisited_nodes\{i} and returning to dest.
    # This is: min over last in unvisited_nodes\{i} of [dp_sub[full_mask_sub][last] + dist(last, dest)]
    # where dp_sub is the DP for the subproblem starting from i.
    
    # This is too complex. Let's just try each first node and solve the subproblem.
    
    best_first = -1
    best_total = INF
    
    for i in range(n):
        # If we choose i first, the cost is dist(current, i) + (cost to visit remaining n-1 nodes from i and return to dest)
        remaining = [unvisited_nodes[j] for j in range(n) if j != i]
        if len(remaining) == 0:
            total = distance_matrix[current_node, unvisited_nodes[i]] + distance_matrix[unvisited_nodes[i], destination_node]
        else:
            # Solve the subproblem: start at unvisited_nodes[i], visit all nodes in 'remaining', return to dest
            # Use DP for the subproblem
            m = len(remaining)
            sub_dp = [[INF] * m for _ in range(1 << m)]
            for j in range(m):
                sub_dp[0][j] = distance_matrix[unvisited_nodes[i], remaining[j]]
            for mask in range(1 << m):
                for last in range(m):
                    if sub_dp[mask][last] == INF:
                        continue
                    if not (mask & (1 << last)):
                        continue
                    for nxt in range(m):
                        if mask & (1 << nxt):
                            continue
                        new_mask = mask | (1 << nxt)
                        cost = sub_dp[mask][last] + distance_matrix[remaining[last], remaining[nxt]]
                        if cost < sub_dp[new_mask][nxt]:
                            sub_dp[new_mask][nxt] = cost
            full_sub_mask = (1 << m) - 1
            sub_cost = INF
            for last in range(m):
                total_sub = sub_dp[full_sub_mask][last] + distance_matrix[remaining[last], destination_node]
                if total_sub < sub_cost:
                    sub_cost = total_sub
            total = distance_matrix[current_node, unvisited_nodes[i]] + sub_cost
        
        if total < best_total:
            best_total = total
            best_first = i
    
    return int(unvisited_nodes[best_first])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check time budget
    if _time_exceeded():
        # Fallback to simple nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited <= 1:
        return int(unvisited_nodes[0]) if n_unvisited == 1 else int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # If we have few enough unvisited nodes, use Held-Karp DP for exact solution
    if n_unvisited <= _held_karp_threshold:
        return _held_karp_dp(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Otherwise, use the penalty-aware heuristic
    # Compute scores for each candidate
    immediate = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[unvisited_nodes, destination_node]
    scores = immediate + 0.5 * dest_dist
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
