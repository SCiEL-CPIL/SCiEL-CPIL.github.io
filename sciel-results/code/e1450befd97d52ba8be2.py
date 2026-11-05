"""Program for Maximum Independent Set via parallel min-degree exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel min-degree exclusion process.
    In each round, vertices with the minimum degree are identified.
    A maximal independent set is greedily selected from these candidates,
    preferring candidates with fewer neighbors within the candidate set.
    Selected vertices and their neighbors are removed from the active set.
    Scores are assigned such that vertices selected in earlier rounds have higher priority,
    and within a round, selected vertices have higher priority than their neighbors.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    
    scores = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    
    round_id = 0
    base_score = n + 1.0
    
    while active.any():
        # Calculate degrees of active vertices considering only active neighbors
        active_mask = active.astype(np.int32)
        deg = (adj * active_mask).sum(axis=1)
        
        # Only consider active vertices
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        active_deg = deg[active_indices]
        min_deg = active_deg.min()
        
        # Find candidates: active vertices with min degree
        candidate_indices = active_indices[active_deg == min_deg]
        
        if len(candidate_indices) == 0:
            break
            
        # Refine sorting: 
        # 1. Primary: Degree (ascending) - already filtered to min_deg, so all equal.
        # 2. Secondary: Number of neighbors within the candidate set (ascending).
        #    Picking a vertex with fewer candidate-neighbors blocks fewer other candidates.
        if len(candidate_indices) > 1:
            cand_sub_adj = adj[np.ix_(candidate_indices, candidate_indices)]
            cand_neighbor_count = cand_sub_adj.sum(axis=1)
            # lexsort: last key is primary. We want degree (all same) then neighbor count.
            # Since degree is constant for candidates, we can just sort by neighbor count.
            # But to be safe/general if we change min_deg logic later:
            sort_keys = np.lexsort((cand_neighbor_count, deg[candidate_indices]))
            sorted_candidates = candidate_indices[sort_keys]
        else:
            sorted_candidates = candidate_indices

        # Greedy Maximal Independent Set on sorted_candidates
        selected_globals = []
        # For efficiency, we can use a mask of 'blocked within round'
        # But K is small, so simple list check is fine.
        # However, vectorized check is faster.
        # Let's stick to the loop but optimize the check.
        for v in sorted_candidates:
            # Check if v is adjacent to any already selected vertex in this round
            if selected_globals:
                # adj[v, selected_globals] is a boolean vector
                if adj[v, selected_globals].any():
                    continue
            selected_globals.append(v)
            
        if not selected_globals:
            break
            
        # Assign scores to selected vertices
        score_val = base_score - round_id
        for v in selected_globals:
            scores[v] = score_val
            
        # Identify neighbors of selected vertices to remove and assign lower scores
        neighbor_mask = adj[selected_globals, :].any(axis=0)
        
        # Update active: remove selected and their neighbors
        active[selected_globals] = False
        active[neighbor_mask] = False
        
        # Assign scores to neighbors (lower than selected, higher than next round)
        neighbors_to_score = np.where(neighbor_mask)[0]
        for u in neighbors_to_score:
            # Ensure we don't overwrite a score if it was somehow set (shouldn't happen)
            if scores[u] == 0.0: 
                scores[u] = score_val - 0.1
        
        round_id += 1
        
    return scores
# EVOLVE-BLOCK-END
