"""Program for Maximum Independent Set via parallel min-degree exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel min-degree exclusion process.
    Simulates a discrete-time exclusion process where vertices with fewest
    active neighbors are selected first. Scores reflect earliest eligibility
    time, capturing dynamic interference patterns.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    # Precompute original degrees for tie-breaking
    orig_deg = adj.sum(axis=1).astype(np.float64)
    
    scores = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    
    round_id = 0
    max_rounds = n  # Safety bound
    
    while active.any() and round_id < max_rounds:
        # Calculate active degrees: count active neighbors for each vertex
        active_deg = (adj & active[None, :]).sum(axis=1)
        
        # Only consider active vertices
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        # Get degrees for active vertices only
        active_deg_vals = active_deg[active_indices]
        min_deg = active_deg_vals.min()
        
        # Find candidates: active vertices with minimum active degree
        candidate_mask = np.zeros(n, dtype=bool)
        candidate_mask[active_indices[active_deg_vals == min_deg]] = True
        
        # For tie-breaking: among candidates, prefer those with lower original degree
        # and lower index (deterministic)
        candidate_indices = np.where(candidate_mask)[0]
        
        # Sort by (original_degree, index) for deterministic tie-breaking
        sort_keys = np.lexsort((candidate_indices, orig_deg[candidate_indices]))
        sorted_candidates = candidate_indices[sort_keys]
        
        # Greedy Maximal Independent Set on sorted_candidates
        # Use a set for O(1) lookup of already-selected vertices
        selected_globals = []
        selected_set = set()
        for v in sorted_candidates:
            # Check if v is adjacent to any already selected vertex in this round
            if selected_globals:
                # Check adjacency to all selected vertices
                if any(adj[v, u] for u in selected_globals):
                    continue
            selected_globals.append(v)
            selected_set.add(v)
            
        if not selected_globals:
            # Fallback: pick the first active vertex
            v = active_indices[0]
            selected_globals = [v]
            selected_set = {v}
        
        # Assign scores: higher score for earlier rounds
        # Selected vertices get high score, their neighbors get slightly lower
        base_score = float(n - round_id)
        
        for v in selected_globals:
            scores[v] = base_score
        
        # Identify neighbors of selected vertices
        if selected_globals:
            neighbor_mask = np.zeros(n, dtype=bool)
            for v in selected_globals:
                neighbor_mask |= adj[v]
            # Exclude the selected vertices themselves from neighbor mask
            neighbor_mask[selected_globals] = False
            
            # Assign lower scores to neighbors (excluded due to interference)
            neighbors_to_score = np.where(neighbor_mask & active)[0]
            for u in neighbors_to_score:
                if scores[u] == 0.0:
                    scores[u] = base_score - 0.5
        
        # Update active: remove selected and their neighbors
        active[selected_globals] = False
        if selected_globals:
            active[neighbor_mask] = False
        
        round_id += 1
    
    # Assign scores to any remaining un-scored vertices (shouldn't happen, but safety)
    unscored = np.where(scores == 0.0)[0]
    if len(unscored) > 0:
        for i, v in enumerate(unscored):
            scores[v] = float(n - round_id) - i * 0.1
    
    return scores
# EVOLVE-BLOCK-END
