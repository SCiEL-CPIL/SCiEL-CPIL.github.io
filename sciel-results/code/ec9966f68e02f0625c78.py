"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Extract coordinates by inverting the distance matrix structure
    # Since we only have distance_matrix, we need to work with distances
    # But for cross product we need coordinates. Let's use a geometric heuristic
    # based on angle from current to destination vs current to candidate.
    
    # Since we don't have explicit coordinates, we'll use a regret-based approach
    # with a geometric penalty for "turning" that approximates cross product behavior.
    
    n_candidates = len(unvisited_nodes)
    if n_candidates == 0:
        return int(current_node)
    
    # Get distances from current to all candidates
    dists_to_candidates = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all candidates
    dest_to_candidates = distance_matrix[destination_node, unvisited_nodes]
    
    # Regret-based scoring: for each candidate, compute how much worse it is
    # compared to the best alternative, weighted by proximity to destination
    
    # Normalize distances
    max_dist = max(np.max(dists_to_candidates), np.max(dest_to_candidates), 1e-10)
    norm_dists = dists_to_candidates / max_dist
    norm_dest_dists = dest_to_candidates / max_dist
    
    # For each candidate, compute regret: difference between 2nd best and best
    # in terms of combined score
    combined_score = 0.6 * norm_dists + 0.4 * norm_dest_dists
    
    # Sort by combined score to get regrets
    sorted_indices = np.argsort(combined_score)
    regrets = np.zeros(n_candidates)
    for i in range(n_candidates):
        if i == 0:
            # Best candidate has regret = 2nd best - best
            if n_candidates > 1:
                regrets[sorted_indices[0]] = combined_score[sorted_indices[1]] - combined_score[sorted_indices[0]]
            else:
                regrets[sorted_indices[0]] = 0
        else:
            # Others have regret = their score - best score
            regrets[sorted_indices[i]] = combined_score[sorted_indices[i]] - combined_score[sorted_indices[0]]
    
    # Add a small geometric heuristic: prefer candidates that create a consistent
    # rotational direction. We approximate this using the relative positioning
    # via distance ratios.
    
    # For each candidate, compute a "turning" measure based on triangle inequality
    # and relative distances
    turn_scores = np.zeros(n_candidates)
    for i, cand in enumerate(unvisited_nodes):
        d_cur_cand = distance_matrix[current_node, cand]
        d_dest_cand = distance_matrix[destination_node, cand]
        d_cur_dest = distance_matrix[current_node, destination_node]
        
        # Compute angle at current using law of cosines
        if d_cur_cand < 1e-10 or d_cur_dest < 1e-10:
            turn_scores[i] = 0
        else:
            cos_angle = (d_cur_cand**2 + d_cur_dest**2 - d_dest_cand**2) / (2 * d_cur_cand * d_cur_dest)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            # Prefer smaller angles (more direct path)
            turn_scores[i] = angle / (np.pi + 1e-10)
    
    # Final score: combine regret (higher is better) and turning penalty (lower is better)
    final_scores = regrets - 0.3 * turn_scores
    
    return int(unvisited_nodes[np.argmax(final_scores)])
# EVOLVE-BLOCK-END
