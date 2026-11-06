    def try_extend(current_colors, current_n, max_time_remaining):
        """Try to add one vertex to the current coloring using efficient backtracking."""
        if time.time() > max_time_remaining:
            return None
        
        # Build adjacency lists for each color: adj[c][i] is set of neighbors of i in color c
        adj = [[set() for _ in range(current_n)] for _ in range(4)]
        idx = 0
        for i in range(current_n):
            for j in range(i + 1, current_n):
                c = current_colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        
        assignment = [-1] * current_n
        # blocked[c] contains vertices that cannot take color c due to previous assignments
        blocked = [set() for _ in range(4)]
        
        def backtrack(pos):
            if time.time() > max_time_remaining:
                return None
            if pos == current_n:
                return assignment[:]
            
            v = pos
            for c in range(4):
                if v not in blocked[c]:
                    # Assign color c to v
                    assignment[v] = c
                    
                    # Save state of blocked[c] for backtracking
                    old_blocked = set(blocked[c])
                    
                    # Update blocked[c]: add neighbors of v in G_c that are > v
                    # These neighbors cannot take color c in future steps
                    for nb in adj[c][v]:
                        if nb > v:
                            blocked[c].add(nb)
                    
                    result = backtrack(pos + 1)
                    if result is not None:
                        return result
                    
                    # Restore state
                    blocked[c] = old_blocked
            assignment[v] = -1
            return None
        
        result = backtrack(0)
        if result is not None:
            return current_colors + result
        return None
    