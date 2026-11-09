"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a hybrid heuristic:
    1. If there are few unvisited nodes (<= 8), use exact DP (Held-Karp) to find the
       optimal completion.
    2. Otherwise, use a lookahead heuristic that balances distance to the candidate
       and the estimated cost of connecting that candidate to the destination.
    
    The lookahead score for candidate c is:
        d(current, c) + alpha * d(c, destination)
    where alpha is a tunable parameter. We use alpha=0.5 as a good default.
    
    Additionally, we consider a 'farthest insertion' style: sometimes picking the node
    that minimizes the total edge cost (current->c + c->destination) is better than
    just nearest neighbor.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen in normal use, but handle gracefully
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # Two choices: go to u0 then u1, or u1 then u0
        # Cost of going to u0 first: d(current, u0) + d(u0, u1) + d(u1, dest)
        # Cost of going to u1 first: d(current, u1) + d(u1, u0) + d(u0, dest)
        u0 = int(unvisited_nodes[0])
        u1 = int(unvisited_nodes[1])
        cost0 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        return u0 if cost0 <= cost1 else u1
    
    if n_unvisited <= 8:
        # Use Held-Karp DP for exact optimal completion
        # State: (mask of visited among unvisited, last node index among unvisited)
        # We want min cost to visit all unvisited and end at destination_node
        m = n_unvisited
        # dp[mask][i] = min cost to visit set mask (subset of unvisited) ending at unvisited[i]
        # Then add d(unvisited[i], destination_node)
        INF = float('inf')
        dp = [[INF] * m for _ in range(1 << m)]
        # Initialize: start from current_node, go to each unvisited[i]
        for i in range(m):
            dp[1 << i][i] = distance_matrix[current_node, unvisited_nodes[i]]
        
        # Fill DP
        for mask in range(1, 1 << m):
            for i in range(m):
                if not (mask & (1 << i)):
                    continue
                if dp[mask][i] == INF:
                    continue
                # Try to go to j not in mask
                rem = ((1 << m) - 1) ^ mask
                if rem == 0:
                    continue
                for j in range(m):
                    if rem & (1 << j):
                        new_mask = mask | (1 << j)
                        cost = dp[mask][i] + distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
                        if cost < dp[new_mask][j]:
                            dp[new_mask][j] = cost
        
        # Find best final step to destination
        full_mask = (1 << m) - 1
        best_cost = INF
        best_i = 0
        for i in range(m):
            if dp[full_mask][i] != INF:
                total = dp[full_mask][i] + distance_matrix[unvisited_nodes[i], destination_node]
                if total < best_cost:
                    best_cost = total
                    best_i = i
        
        return int(unvisited_nodes[best_i])
    
    # For larger n_unvisited, use lookahead heuristic
    # Score each candidate c: d(current, c) + alpha * d(c, destination)
    # Also consider a second score: d(current, c) + d(c, destination) (alpha=1)
    # We blend: use alpha=0.5
    
    # Vectorized computation
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Primary score: nearest neighbor with lookahead
    alpha = 0.5
    scores = dists_from_current + alpha * dists_to_dest
    
    # Also compute a "regret" score: how much worse is picking c vs. the best alternative
    # This is a second-order heuristic. For simplicity, use the primary score.
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
