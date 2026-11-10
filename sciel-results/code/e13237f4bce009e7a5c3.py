        best_edges = current_edges
        for _ in range(10): # Try 10 random orders
            random.shuffle(current_edges)
            i = 0
            while i < len(current_edges):
                new_edges = current_edges[:i] + current_edges[i+1:]
                if is_not_2_colorable_fast(n, new_edges):
                    current_edges = new_edges
                    improved = True
                else:
                    i += 1
            if len(current_edges) < len(best_edges):
                best_edges = current_edges
