<<<<<<< SEARCH
    def backtrack():
        # Find next unassigned vertex
        next_v = -1
        for v in range(n):
            if colors[v] == -1:
                next_v = v
                break
        
        if next_v == -1:
            return True
        
        # Save state to restore later
        saved_colors = colors[:]
        
        # Try 0
        colors[next_v] = 0
        if propagate() and backtrack():
            return True
        
        # Try 1
        colors[next_v] = 1
        if propagate() and backtrack():
            return True
        
        # Restore state
        colors[:] = saved_colors
        return False
=======
    def backtrack():
        # Find next unassigned vertex
        next_v = -1
        for v in range(n):
            if colors[v] == -1:
                next_v = v
                break
        
        if next_v == -1:
            return True
        
        # Save state to restore later
        saved_colors = colors[:]
        
        # Try 0
        colors[next_v] = 0
        if propagate():
            if backtrack():
                return True
            # Restore state before trying next branch
            colors[:] = saved_colors
        
        # Try 1
        colors[next_v] = 1
        if propagate():
            if backtrack():
                return True
            # Restore state
            colors[:] = saved_colors
        
        # Restore state (if both failed, though colors is already restored)
        colors[:] = saved_colors
        return False
>>>>>>> REPLACE
   