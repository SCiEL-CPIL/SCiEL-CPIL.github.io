    def solve_instance(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        compat = build_compatibility_graph(edges, adj)
        
        remaining = set(range(m))
        colors = [None] * m
        current_color = 0
        
        while remaining:
            if time.time() > deadline:
                for v in remaining:
                    colors[v] = current_color
                    current_color += 1
                break
            
            # Find vertex with min degree in remaining
            min_deg = float('inf')
            target_v = None
            for v in remaining:
                deg = len(remaining & compat[v])
                if deg < min_deg:
                    min_deg = deg
                    target_v = v
            
            # Find Max Clique in neighbors of target_v within remaining
            # The clique must include target_v.
            # So we find Max Clique in (remaining ∩ compat[target_v])
            # and add target_v to it.
            
            candidates_for_clique = remaining & compat[target_v]
            
            # Use BK to find max clique in candidates_for_clique
            # We pass P = candidates_for_clique, X = empty
            # R = empty (we will add target_v later)
            
            clique_body = max_clique_bron_kerbosch(set(), candidates_for_clique, set(), compat, deadline)
            
            # The full clique is target_v + clique_body
            full_clique = [target_v] + clique_body
            
            # Assign color
            for v in full_clique:
                colors[v] = current_color
                remaining.remove(v)
            
            current_color += 1
            
        return colors
