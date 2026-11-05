<<<<<<< SEARCH
    def multi_restart_greedy(edges, conflicts, n_edges, deadline, n_restarts=5):
        """
        Multi-restart greedy with random ordering to find good initial solutions.
        """
        best_color = None
        best_n_colors = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Random ordering
            order = list(range(n_edges))
            random.shuffle(order)
            
            color, n_colors = greedy_color_fast(order, conflicts, n_edges)
            
            if n_colors < best_n_colors:
                best_n_colors = n_colors
                best_color = color[:]
        
        if best_color is None:
            order = list(range(n_edges))
            best_color, best_n_colors = greedy_color_fast(order, conflicts, n_edges)
        
        return best_color, best_n_colors
=======
    def multi_restart_greedy(edges, conflicts, n_edges, deadline, n_restarts=5):
        """
        Multi-restart greedy with mixed ordering strategies to find good initial solutions.
        """
        best_color = None
        best_n_colors = float('inf')
        
        # First try DSATUR ordering (deterministic, often good)
        dsatur = dsatur_order(conflicts, n_edges)
        color, n_colors = greedy_color_fast(dsatur, conflicts, n_edges)
        best_color = color[:]
        best_n_colors = n_colors
        
        # Then try random orderings
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Random ordering
            order = list(range(n_edges))
            random.shuffle(order)
            
            color, n_colors = greedy_color_fast(order, conflicts, n_edges)
            
            if n_colors < best_n_colors:
                best_n_colors = n_colors
                best_color = color[:]
        
        return best_color, best_n_colors
>>>>>>> REPLACE

