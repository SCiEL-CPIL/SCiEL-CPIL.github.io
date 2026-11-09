import numpy as np
import time

_start_time = None


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a connectivity-aware heuristic that approximates
    the minimum spanning tree cost of the remaining unvisited nodes plus the
    connection to current and destination.
    
    For each candidate c, we compute:
      score(c) = dist(current, c) + dist(c, dest) + MST_cost(unvisited_nodes \\ {c})
    
    where MST_cost is the total weight of the minimum spanning tree on the
    remaining unvisited nodes (excluding c). This captures global connectivity:
    choosing a candidate that leaves the rest of the unvisited set poorly
    connected (high MST cost) is penalized.
    
    For small n, we use exact or near-exact evaluations.
    """
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 1.0:
        if len(unvisited_nodes) == 0:
            return int(destination_node)
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        # Pick the candidate that minimizes dist(current, c) + dist(c, dest)
        cands = unvisited_nodes
        scores = distance_matrix[current_node, cands] + distance_matrix[cands, destination_node]
        return int(cands[np.argmin(scores)])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(candidates[0])

    # For n == 2, exact evaluation
    if n == 2:
        c0 = candidates[0]
        c1 = candidates[1]
        cost0 = distance_matrix[current_node, c0] + distance_matrix[c0, c1] + distance_matrix[c1, destination_node]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c0] + distance_matrix[c0, destination_node]
        return int(c0) if cost0 < cost1 else int(c1)

    # For n >= 3, use MST-based heuristic
    # We'll compute for each candidate c:
    #   score(c) = dist(current, c) + dist(c, dest) + MST(remaining after removing c)
    
    cand_idx = candidates.astype(int)
    N = n
    
    # Precompute distance matrix among candidates
    cand_dist = distance_matrix[np.ix_(cand_idx, cand_idx)]
    
    # For efficiency with moderate N, we can use a vectorized approach.
    # For each candidate i, we need MST of the subgraph excluding i.
    # This is expensive for large N, so let's cap the MST computation.
    
    # If N is large, use a simpler approximation: 
    #   score(c) = dist(current, c) + dist(c, dest) + sum of min-edges in remaining
    # Actually, let's try to be smarter.
    
    # For N up to ~50, we can compute MST for each candidate removal.
    # But 50 MSTs on 49 nodes each might be slow. Let's use Prim's algorithm efficiently.
    
    # Alternative: use the fact that removing one node from an MST changes the MST
    # by at most one edge. But that's complex. Let's just do it directly for small N.
    
    if N <= 30:
        # Compute MST cost for each candidate removal
        scores = np.empty(N)
        for i in range(N):
            # Remove candidate i
            remaining = np.delete(cand_idx, i)
            rem_dist = distance_matrix[np.ix_(remaining, remaining)]
            
            # Compute MST weight using Prim's algorithm
            m = len(remaining)
            if m <= 1:
                mst_cost = 0.0
            else:
                # Prim's algorithm
                in_mst = np.zeros(m, dtype=bool)
                in_mst[0] = True
                mst_weight = 0.0
                min_edge = rem_dist[0].copy()
                
                for _ in range(m - 1):
                    # Find the minimum edge connecting an in_mst node to an out_mst node
                    candidates_edges = min_edge.copy()
                    candidates_edges[in_mst] = np.inf
                    j = np.argmin(candidates_edges)
                    mst_weight += min_edge[j]
                    in_mst[j] = True
                    # Update min_edge
                    update = rem_dist[j]
                    min_edge = np.minimum(min_edge, update)
                    min_edge[in_mst] = np.inf
                mst_cost = mst_weight
            
            scores[i] = distance_matrix[current_node, cand_idx[i]] + distance_matrix[cand_idx[i], destination_node] + mst_cost
        
        idx = np.argmin(scores)
        return int(candidates[idx])
    else:
        # For larger N, use a cheaper approximation:
        # score(c) = dist(current, c) + dist(c, dest) + sum of (second-min distance from each node in remaining)
        # This approximates the MST cost as a sum of local connection costs.
        
        d_curr = distance_matrix[current_node, candidates]
        d_dest = distance_matrix[candidates, destination_node]
        
        # For each candidate, compute the sum of min-distances from other candidates to their nearest neighbor (excluding self)
        # This is a lower bound on MST cost.
        
        # To make this vectorized:
        # For each i, we want sum over j != i of min_{k != i, k != j} dist(j, k)
        # This is expensive to compute exactly for all i.
        
        # Simpler: use the sum of the smallest edge incident to each node in the remaining set.
        # For node j in remaining (j != i), the min incident edge is min_{k in remaining, k != j} dist(j, k).
        
        # Let's compute for each candidate i, the MST approximation as:
        #   sum_{j in remaining} min_{k in remaining, k != j} dist(j, k)
        # But this double-counts edges, so it's 2 * (sum of min-incident edges) / something.
        
        # Actually, a simpler and still effective heuristic:
        # Use the sum of the two smallest edges from each node as a proxy for connectivity.
        
        # Let's just use: score(c) = d_curr[c] + d_dest[c] + alpha * (average min-distance from c to remaining)
        # where alpha is a tuning parameter.
        
        # Compute for each candidate i: min distance from i to any other candidate (already computed)
        cand_dist_with_self = cand_dist.copy()
        np.fill_diagonal(cand_dist_with_self, np.inf)
        min_dist_to_other = np.min(cand_dist_with_self, axis=1)
        
        # For each candidate i, the "cost of leaving i out" can be approximated by
        # how well-connected the rest is. A simple proxy: the average min_dist_to_other of the remaining nodes.
        
        # sum_min = sum of min_dist_to_other for all candidates
        total_min = np.sum(min_dist_to_other)
        
        # For candidate i, the sum for remaining = total_min - min_dist_to_other[i]
        # But min_dist_to_other[i] was computed with i in the set, so it includes distance to other candidates.
        # When i is removed, the min_dist for other nodes might change slightly, but this is a good approximation.
        
        remaining_min_sum = total_min - min_dist_to_other
        
        # Normalize: divide by (N-1) to get average
        avg_remaining_min = remaining_min_sum / (N - 1)
        
        # Combine
        w = 1.0 / N
        scores = d_curr + d_dest + (1.0 - w) * avg_remaining_min
        
        idx = np.argmin(scores)
        return int(candidates[idx])
