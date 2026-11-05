        # Inside local search loop:
        if not has_edge:
            # Add edge (u, v)
            # Check if N(u) & N(v) has an edge
            if has_edge_in_set(best_adj[u] & best_adj[v], best_adj):
                continue # Would create K4
            # Safe to add
