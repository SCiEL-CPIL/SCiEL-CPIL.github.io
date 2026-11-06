import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 145  # Safety margin for 150s limit

def _held_karp_next(current, unvisited, destination, dist_matrix):
    """
    Exact solution for the next node when the number of unvisited nodes is small.
    Uses Held-Karp DP to find the shortest path from 'current' visiting all 'unvisited' and ending at 'destination'.
    Returns the first node in that optimal path.
    """
    n = len(unvisited)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited[0])
    
    # dp[mask][i] = min cost to start at current, visit set 'mask', and end at node 'i' (index in unvisited)
    size = 1 << n
    INF = float('inf')
    dp = [[INF] * n for _ in range(size)]
    
    # Base case: visit only node i
    for i in range(n):
        dp[1 << i][i] = dist_matrix[current, unvisited[i]]
        
    # Fill DP
    for mask in range(size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == INF:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist_matrix[unvisited[i], unvisited[j]]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    
    # Find the best full path ending at destination
    full_mask = size - 1
    best_total = INF
    best_first_node = -1
    
    # For each possible first node k, compute the exact cost
    # We do this by checking which first node leads to the optimal solution
    # Actually, let's just find the best overall cost and backtrack
    
    # Find best end node
    best_end = -1
    best_cost = INF
    for i in range(n):
        total = dp[full_mask][i] + dist_matrix[unvisited[i], destination]
        if total < best_cost:
            best_cost = total
            best_end = i
            
    # Backtrace to find the first node
    # We need parent pointers. Let's reconstruct by checking predecessors.
    # Since we want the first node, we can just look at the base cases that lead to the optimal path.
    # However, standard backtracking requires parent pointers.
    # Given the constraints (n <= 15), we can afford to store parents or just re-evaluate.
    # Let's use a simpler approach for n <= 15:
    # Compute the cost for each candidate first node 'k' explicitly using a recursive DP or by modifying the DP to track first nodes.
    # Actually, the most robust way for small n is to just compute the min cost for each first node.
    
    # Cost if first node is k:
    # dist(current, k) + min_{perm of rest} (dist(k, p1) + ... + dist(pk, destination))
    # This is equivalent to: dist(current, k) + (optimal cost to visit all nodes in unvisited\{k} starting from k and ending at destination)
    
    # Let's precompute a DP for the "suffix" cost.
    # suffix_dp[mask][i] = min cost to visit set 'mask' (subset of unvisited) starting from 'i' (which is NOT in mask? No, i is the start, mask is the set to visit).
    # Let's define: g(mask, i) = min cost to visit all nodes in 'mask' starting from node 'i' (where i is not in mask, or i is the previous node).
    # Here, i is the current position, mask is the set of unvisited nodes to be visited.
    # We want g(unvisited\{k}, k) for each k.
    
    # This is essentially the same DP but with different boundary conditions.
    # Let's just use the parent pointer approach for simplicity, as n is small.
    
    parent = [[-1] * n for _ in range(size)]
    
    # Re-run DP with parent tracking (it's fast for n<=15)
    dp = [[INF] * n for _ in range(size)]
    for i in range(n):
        dp[1 << i][i] = dist_matrix[current, unvisited[i]]
        
    for mask in range(size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == INF:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist_matrix[unvisited[i], unvisited[j]]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    parent[new_mask][j] = i
                    
    # Find best end node again (dp was reset)
    best_end = -1
    best_cost = INF
    for i in range(n):
        total = dp[full_mask][i] + dist_matrix[unvisited[i], destination]
        if total < best_cost:
            best_cost = total
            best_end = i
            
    # Backtrace
    cur_mask = full_mask
    cur_node = best_end
    while (cur_mask & (cur_mask - 1)) != 0:
        prev_node = parent[cur_mask][cur_node]
        cur_mask = cur_mask & ~(1 << cur_node)
        cur_node = prev_node
        
    return int(unvisited[cur_node])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check
    if time.time() > _DEADLINE - 1.0:
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    # Use exact Held-Karp for small subproblems
    if n_unvisited <= 15:
        return _held_karp_next(current_node, unvisited_nodes, destination_node, distance_matrix)
    
    # For larger n_unvisited, use a hybrid approach
    # 1. Score all nodes using a fast heuristic
    # 2. Select top K candidates
    # 3. Apply Held-Karp on the selected candidates + neighbors? 
    # No, the directive says: "Replace the sequential greedy selection with a dynamic programming search (Held-Karp) over the last 15 unvisited nodes to compute the exact optimal suffix, while using a fast insertion heuristic for the initial nodes to ensure the exact search remains within the time limit."
    
    # This implies that for the *initial* part of the construction (when n_unvisited is large), we should use a fast heuristic.
    # But the function `select_next_node` is called at *every* step.
    # So, when n_unvisited is large, we are in the "initial" phase.
    # When n_unvisited is small (<= 15), we are in the "suffix" phase.
    
    # The current code already does this:
    # if n_unvisited <= 15: return _held_karp_next(...)
    # else: use heuristic
    
    # The improvement should be in the heuristic used for the "initial" phase (n_unvisited > 15).
    # The current heuristic is a 2-step lookahead + DP on top K.
    # Let's improve the heuristic for n_unvisited > 15.
    
    # Strategy for n_unvisited > 15:
    # Use a fast insertion heuristic or a smarter greedy method.
    # The current method uses a 2-step lookahead score.
    # Let's try a 3-step lookahead or a combination.
    
    # Compute distances
    dist_current_to_v = distance_matrix[current_node, unvisited_nodes]
    dist_v_to_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    dist_v_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 2-step lookahead
    M_temp = dist_v_to_unvisited + dist_v_to_dest[None, :]
    np.fill_diagonal(M_temp, np.inf)
    best_cost_2step = np.min(M_temp, axis=1)
    
    # 3-step lookahead
    # For each v, find min over u1, u2 (u1 != v, u2 != v, u2 != u1) of dist(v, u1) + dist(u1, u2) + dist(u2, dest)
    # This is O(N^3) which might be too slow for N=50 (125k ops, okay).
    # But we can vectorize it.
    
    # Let's stick to a refined 2-step lookahead with a penalty for ignoring close nodes.
    # Or, use the top K candidates and run a small DP on them, but include the cost of the remaining nodes as an estimate.
    
    # The current code selects top K=10 and runs DP on them.
    # But the DP only considers visiting those K nodes and then going to destination.
    # It ignores the cost of visiting the other (n_unvisited - K) nodes.
    # This is a flaw. The DP should estimate the cost of the remaining nodes.
    
    # Let's fix this:
    # For each candidate first node v (from top K), compute:
    # dist(current, v) + (exact cost to visit K-1 other selected nodes and go to 'virtual' destination)
    # where the 'virtual' destination cost includes the estimated cost of visiting the remaining nodes.
    
    # Actually, a simpler and more effective approach:
    # Use the top K nodes (K=15) and run Held-Karp on them.
    # But modify the DP to include an estimate for the remaining nodes.
    # This is complex.
    
    # Alternative:
    # Use a beam search with width 10 and depth 3 for the initial phase.
    # Or, just use a better greedy score.
    
    # Let's try: Score = dist(current, v) + 0.5 * min_{u != v} (dist(v, u) + dist(u, dest))
    # This is the 2-step lookahead with a weight.
    
    # Let's try increasing K to 12 for the DP refinement, but include the remaining cost in the DP.
    # We can approximate the remaining cost by assuming the remaining nodes are visited by nearest neighbor from the last selected node.
    
    # Let's implement a DP that includes the remaining cost estimate.
    
    K = 12
    sorted_indices = np.argsort(dist_current_to_v + best_cost_2step)
    refine_indices = sorted_indices[:K]
    refine_nodes = unvisited_nodes[refine_indices]
    
    k = K
    INF = float('inf')
    dp = np.full((1 << k, k), INF, dtype=np.float64)
    
    for i in range(k):
        dp[1 << i, i] = distance_matrix[current_node, refine_nodes[i]]
        
    for mask in range(1, 1 << k):
        for last in range(k):
            if dp[mask, last] == INF:
                continue
            if not (mask & (1 << last)):
                continue
            for nxt in range(k):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                cost = dp[mask, last] + distance_matrix[refine_nodes[last], refine_nodes[nxt]]
                if cost < dp[new_mask, nxt]:
                    dp[new_mask, nxt] = cost
    
    full_mask = (1 << k) - 1
    
    # Estimate the cost of the remaining nodes (n_unvisited - k)
    # We assume they are visited by nearest neighbor starting from the last selected node and ending at destination.
    # We can precompute this estimate for each possible last selected node.
    
    remaining_nodes = np.setdiff1d(unvisited_nodes, refine_nodes)
    
    def estimate_remaining_cost_from(last_node):
        if len(remaining_nodes) == 0:
            return distance_matrix[last_node, destination_node]
        
        remaining = list(remaining_nodes)
        total_cost = 0.0
        curr = last_node
        
        while remaining:
            dists = distance_matrix[curr, remaining]
            min_idx = np.argmin(dists)
            next_node = remaining[min_idx]
            total_cost += dists[min_idx]
            curr = next_node
            remaining.pop(min_idx)
            
        total_cost += distance_matrix[curr, destination_node]
        return total_cost
    
    best_total = INF
    best_last = -1
    for last in range(k):
        total = dp[full_mask, last] + estimate_remaining_cost_from(refine_nodes[last])
        if total < best_total:
            best_total = total
            best_last = last
    
    # Backtrack to find the first node
    path = []
    mask = full_mask
    last = best_last
    while mask != 0:
        path.append(last)
        prev_mask = mask & ~(1 << last)
        if prev_mask == 0:
            break
        best_prev_cost = INF
        best_prev = -1
        for prev in range(k):
            if not (prev_mask & (1 << prev)):
                continue
            cost = dp[prev_mask, prev] + distance_matrix[refine_nodes[prev], refine_nodes[last]]
            if cost < best_prev_cost - 1e-9:
                best_prev_cost = cost
                best_prev = prev
        if best_prev == -1:
            break
        last = best_prev
        mask = prev_mask
    
    path.reverse()
    if len(path) > 0:
        first_local_idx = path[0]
        return int(refine_nodes[first_local_idx])
    else:
        return int(unvisited_nodes[refine_indices[0]])
