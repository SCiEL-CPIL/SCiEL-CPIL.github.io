    def backtrack():
        unassigned = [v for v in range(n) if colors[v] == 0]
        if not unassigned:
            return True
            
        v = unassigned[0]
        saved_colors = colors[:]
        
        # Try 1
        colors[v] = 1
        if propagate():
            if backtrack():
                return True
        
        # Restore
        colors[:] = saved_colors
        
        # Try 2
        colors[v] = 2
        if propagate():
            if backtrack():
                return True
        
        # Restore (not strictly necessary if we return False, but good practice)
        colors[:] = saved_colors
        
        return False
