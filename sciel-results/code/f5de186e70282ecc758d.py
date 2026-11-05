def find_i6(n, adj):
    """Finds an independent set of size 6. Returns set of 6 vertices or None."""
    
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        
        if len(candidates) < 6 - len(current_set):
            return None
            
        # Heuristic: Pick vertex with max degree in the induced subgraph on candidates
        # This minimizes the number of non-neighbors (next candidates)
        best_v = -1
        max_deg = -1
        for v in candidates:
            deg = 0
            for w in candidates:
                if v != w and (adj[v] & (1 << w)):
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                best_v = v
                
        if best_v == -1:
            return None
            
        v = best_v
        # Generate new candidates: non-neighbors of v in candidates, excluding v
        new_candidates = []
        for w in candidates:
            if w != v and not (adj[v] & (1 << w)):
                new_candidates.append(w)
                
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        return None

    return backtrack(list(range(n)), [])
