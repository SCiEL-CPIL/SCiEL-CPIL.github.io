    adj_sets = adj # Keep the set version
    adj_lists = [list(s) for s in adj] # For iteration
    
    # ... inside solve ...
    new_candidates = [v for v in remaining if v not in adj_sets[u]]
