# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.sparse import csr_matrix


def _compute_mst_cost(nodes, distance_matrix):
    """
    Compute the total weight of the Minimum Spanning Tree (MST) of the given nodes
    using the provided distance matrix. This serves as a lower bound/estimate for
    the cost to visit all remaining nodes and return.
    """
    n = len(nodes)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    
    # Extract submatrix for unvisited nodes
    sub_matrix = distance_matrix[np.ix_(nodes, nodes)]
    
    # Build sparse graph for MST computation
    # We need to convert the dense matrix to a sparse format suitable for scipy
    # To avoid self-loops (diagonal), we can set diagonal to inf or just ignore them in MST
    # scipy's minimum_spanning_tree works on sparse graphs
    # Let's create a sparse matrix from the dense sub_matrix
    # Note: distance_matrix is symmetric, so we can use it directly.
    
    # For efficiency with small n (unvisited nodes decrease over time), dense is fine, 
    # but scipy.sparse.csgraph functions expect sparse or dense arrays.
    # Let's try using dense array directly if n is small enough, otherwise sparse.
    # Actually, minimum_spanning_tree accepts a dense array as well.
    
    try:
        mst = minimum_spanning_tree(sub_matrix)
        mst_weight = np.sum(mst.data) if mst.nnz > 0 else 0.0
        return float(mst_weight)
    except Exception:
        # Fallback: simple sum of smallest n-1 edges? Or just return a large penalty
        # For robustness, if MST fails, return a large value
        return float('inf')


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that combines
    immediate distance with an MST-based estimate of the residual tour cost.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen if called correctly, but handle gracefully
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Pre-compute MST cost for the current set of unvisited nodes
    # This cost represents the estimated cost to connect all unvisited nodes together.
    # When we move to a candidate node 'u', the new set of unvisited nodes will be 
    # unvisited_nodes \ {u}. We need to estimate the cost for that new set.
    
    # To optimize, we can't recompute MST for every candidate if n is large, 
    # but n starts at 49 and decreases. MST on 49x49 is fast.
    # Let's compute the MST cost for the *current* unvisited set first.
    # Then for each candidate, we approximate the new MST cost.
    
    # A simpler and effective approach:
    # Score(u) = dist(current, u) + dist(u, destination) + 2 * (MST(unvisited \ {u}) / 2) ?
    # Actually, the residual tour cost after visiting u is roughly the cost to complete the tour from u back to destination,
    # visiting all other unvisited nodes.
    # A common lower bound is MST(unvisited) + min(dist(u, v) for v in unvisited) ... this is getting complex.
    
    # Let's stick to the directive: 
    # "selecting the next node that minimizes the sum of the immediate distance and this MST-based residual estimate"
    # The residual estimate for the remaining nodes after choosing u is the MST of (unvisited \ {u}).
    
    # Optimization: 
    # Computing MST for each of the n_unvisited candidates is O(n * E log V) which might be slow for n=49.
    # 49 * (MST on 48 nodes) is acceptable in Python/SciPy for 120s limit? 
    # MST on 48x48 dense matrix is very fast. Let's try it.
    
    best_score = float('inf')
    best_node = -1
    
    # We can pre-compute the current MST cost to use as a baseline or for pruning?
    # For now, direct computation per candidate.
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost: distance from current to candidate
        immediate_dist = distance_matrix[current_node, candidate]
        
        # Residual cost estimate:
        # After visiting candidate, the remaining unvisited nodes are unvisited_nodes without candidate.
        # We estimate the cost to finish the tour from candidate to destination, visiting all remaining nodes.
        # A reasonable lower bound/estimate is the MST of the remaining nodes plus the connection costs.
        # However, the directive says "MST of the unvisited nodes to estimate the residual tour cost".
        # Let's interpret "residual tour cost" as the cost to visit all *other* unvisited nodes and return to destination.
        
        # Let's compute the MST of the set of unvisited nodes *excluding* the candidate.
        remaining_indices = [j for j in range(n_unvisited) if j != i]
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            # Only candidate left, we already handled n_unvisited==1, so this shouldn't happen in the loop
            residual_est = 0.0
        else:
            residual_est = _compute_mst_cost(remaining_nodes, distance_matrix)
            
            # The residual tour also needs to connect the candidate to the rest and eventually to the destination.
            # A better estimate: 
            # Cost to go from candidate to the "cluster" of remaining nodes + MST of remaining + Cost to go from "cluster" to destination.
            # Simplest robust estimate: 
            # residual_est = MST(remaining) + min_dist(candidate, remaining) + min_dist(destination, remaining) ?
            # Or even simpler: just use MST(remaining) as a proxy for the complexity of the remaining sub-tour.
            
            # Let's use a weighted combination:
            # The final tour segment will be: current -> candidate -> ... -> destination
            # The cost of the sub-tour visiting remaining nodes starting/ending near candidate/destination is at least MST(remaining).
            # Let's add the distance from candidate to the nearest remaining node and from destination to the nearest remaining node to account for connections.
            if len(remaining_nodes) > 0:
                min_dist_cand_rem = np.min(distance_matrix[candidate, remaining_nodes])
                min_dist_dest_rem = np.min(distance_matrix[destination_node, remaining_nodes])
                residual_est += min_dist_cand_rem + min_dist_dest_rem
            else:
                # No remaining nodes, just go to destination
                residual_est = distance_matrix[candidate, destination_node]
        
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
