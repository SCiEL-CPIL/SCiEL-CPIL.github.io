"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a traveling salesman tour using a 2-step lookahead heuristic.
    
    This structural change moves from a 1-step weighted sum (dist_to_current + lambda * dist_to_dest)
    to a 2-step lookahead that evaluates the cost of moving to a candidate i and then to the
    best subsequent node j.
    
    Score(i) = dist(current, i) + min_{j != i} (dist(i, j) + dist(j, destination))
    
    This heuristic:
    1. Avoids isolated nodes (nodes with no close neighbors) by penalizing high min_{j} dist(i, j).
    2. For the last two nodes, it provides the exact optimal ordering.
    3. Reduces tour crossings by ensuring the path continues into a dense cluster.
    
    Complexity: O(N^2) per step, which is efficient for N=50.
    """
    # Safety check: if we are close to the deadline, fall back to a very fast greedy
    # (though for N=50, the O(N^2) method is likely fast enough, this is a safeguard
    # against unexpected overhead in the evaluator environment).
    if time.time() > _DEADLINE - 5.0:
        # Fallback: Pure nearest neighbor to current
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])

    n = len(unvisited_nodes)
    if n == 0:
        return int(current_node)
    if n == 1:
        return int(unvisited_nodes[0])
    
    u = unvisited_nodes
    d_mat = distance_matrix
    
    # 1. Distances from current to all unvisited
    dist_c_u = d_mat[current_node, u]
    
    # 2. Distances between all unvisited nodes (N x N)
    # Using advanced indexing for vectorized distance matrix slice
    dist_u_u = d_mat[u[:, np.newaxis], u[np.newaxis, :]]
    
    # 3. Distances from unvisited to destination
    dist_u_d = d_mat[u, destination_node]
    
    # 4. Calculate the "next step" cost for each candidate i
    # We want to find min_j (dist(i, j) + dist(j, destination))
    # This is equivalent to min_j (dist_u_u[i, j] + dist_u_d[j])
    
    # Construct the cost matrix for the next step
    # next_cost[i, j] = dist(i, j) + dist(j, destination)
    next_cost = dist_u_u + dist_u_d[np.newaxis, :]
    
    # Mask the diagonal (j == i) with infinity, as we cannot stay at the same node
    np.fill_diagonal(next_cost, np.inf)
    
    # Find the minimum next-step cost for each candidate i
    min_next_cost = np.min(next_cost, axis=1)
    
    # 5. Total score: cost to get to i + estimated cost from i to finish
    total_score = dist_c_u + min_next_cost
    
    # Select the candidate with the minimum total score
    best_idx = np.argmin(total_score)
    
    return int(u[best_idx])
# EVOLVE-BLOCK-END
