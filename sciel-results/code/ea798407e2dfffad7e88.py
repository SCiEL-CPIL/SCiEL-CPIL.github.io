# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for reference tour successors, keyed by hash of distance matrix
_ref_tour_cache = {}

def _compute_reference_successors(distance_matrix):
    """
    Compute a near-optimal reference tour using NN + 2-Opt, and return
    the successor mapping (for each node, the next node in the reference tour).
    This is cached so it's computed only once per unique distance matrix.
    """
    n = distance_matrix.shape[0]
    
    # Use a simple hash based on matrix size and a few sampled values
    # to avoid expensive full-matrix hashing every call
    key = (n, tuple(distance_matrix[0, :5]), tuple(distance_matrix[-1, -5:]), 
           distance_matrix[0,0], distance_matrix[1,1] if n > 1 else 0)
    
    if key in _ref_tour_cache:
        return _ref_tour_cache[key]
    
    # Build reference tour using NN + 2-Opt
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    
    for _ in range(n - 1):
        current = tour[-1]
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
    
    # 2-Opt improvement
    def tour_length(t):
        total = 0.0
        for i in range(len(t)):
            total += distance_matrix[t[i], t[(i+1) % len(t)]]
        return total
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                # Check 2-opt move
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[(j+1) % n]
                # Gain from reversing segment i..j
                old_len = distance_matrix[a, b] + distance_matrix[c, d]
                new_len = distance_matrix[a, c] + distance_matrix[b, d]
                if new_len < old_len - 1e-10:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    
    # Build successor mapping
    successors = np.zeros(n, dtype=int)
    for i in range(n):
        successors[tour[i]] = tour[(i+1) % n]
    
    _ref_tour_cache[key] = successors
    return successors


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node by combining:
    1. Reference tour successor (from NN+2-Opt) - avoids greedy myopia
    2. Local geometric scoring as tiebreaker/refinement
    
    Strategy: If the reference tour's successor for current_node is in unvisited,
    prefer it. Otherwise, use a combined score that balances distance to destination
    and local geometric fit.
    """
    if len(unvisited_nodes) == 0:
        return current_node
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Get reference tour successors
    ref_successors = _compute_reference_successors(distance_matrix)
    
    # Check if reference successor is available
    ref_next = ref_successors[current_node]
    if ref_next in unvisited_nodes:
        # Strong preference for reference tour successor
        # But verify it's not catastrophically bad compared to alternatives
        ref_dist = distance_matrix[current_node, ref_next]
        
        # Quick check: if the nearest neighbor is much closer AND the reference
        # successor is very far, consider a blend
        dists_to_unvisited = distance_matrix[current_node, unvisited_nodes]
        min_dist = np.min(dists_to_unvisited)
        
        # If reference successor is within 1.5x of the minimum, take it
        if ref_dist <= 1.5 * min_dist or len(unvisited_nodes) <= 3:
            return int(ref_next)
    
    # Fallback: use a combined scoring function
    # Score = alpha * dist_to_candidate + beta * dist_candidate_to_destination
    # + gamma * (angle penalty for turning)
    
    candidates = unvisited_nodes
    dists_from_current = distance_matrix[current_node, candidates]
    dists_to_dest = distance_matrix[candidates, destination_node]
    
    # Compute angle penalty: how much does going current->candidate->destination
    # deviate from a straight line? Lower is better (straighter is better)
    # Vector from current to candidate, and from candidate to destination
    # The angle between them: if small, it's a smooth path
    
    # Get coordinates implicitly from distance matrix? No, we only have distances.
    # Use a proxy: the triangle inequality residual
    # dist(current, dest) vs dist(current, cand) + dist(cand, dest)
    # If cand is "on the way", dist(c,dest) ≈ dist(c,cand) + dist(cand,dest)
    
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    # "Straightness" score: lower means more direct
    # For each candidate, compute how much extra distance it adds
    # compared to going directly
    extra_dist = dists_from_current + dists_to_dest - dist_current_to_dest
    # Normalize
    max_extra = np.max(extra_dist) if np.max(extra_dist) > 0 else 1.0
    straightness = extra_dist / max_extra
    
    # Combined score: prefer low distance from current, low extra distance
    # Weights: emphasize going toward destination
    alpha = 0.4  # weight for distance from current
    beta = 0.3   # weight for distance to destination  
    gamma = 0.3  # weight for straightness
    
    # Normalize each component
    max_d1 = np.max(dists_from_current) if np.max(dists_from_current) > 0 else 1.0
    max_d2 = np.max(dists_to_dest) if np.max(dists_to_dest) > 0 else 1.0
    
    norm_d1 = dists_from_current / max_d1
    norm_d2 = dists_to_dest / max_d2
    
    combined_score = alpha * norm_d1 + beta * norm_d2 + gamma * straightness
    
    best_idx = int(np.argmin(combined_score))
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
