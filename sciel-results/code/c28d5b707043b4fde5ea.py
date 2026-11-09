def construct():
        from random import Random
        
        def instance(n, k, p, seed):
            rng = Random(seed)
            # Generate edges
            edges = set()
            # Clique part C
            for u in range(k):
                for v in range(u + 1, k):
                    edges.add((u, v))
            # Cross edges
            for v in range(k, n):
                for u in range(k):
                    if rng.random() < p:
                        edges.add((u, v))
            
            # Build cliques
            cliques = []
            
            # 1. Mixed cliques: [u, v] for each cross edge (u, v)
            # We need to iterate cross edges specifically.
            # To avoid regenerating or checking set membership for every pair,
            # we can just reconstruct the cross edges or filter.
            
            # Better: Regenerate cross edges logic or store them.
            # The instance function in the prompt returns sorted edges.
            # We can just iterate the edges.
            
            # Let's stick to the logic:
            # Cross edges are (u, v) where u < k and v >= k.
            # Internal edges are (u, v) where u, v < k.
            
            cross_edges = []
            internal_edges = []
            
            for e in edges:
                u, v = e
                if u < k and v >= k:
                    cross_edges.append((u, v))
                elif v < k and u >= k: # Should not happen if u<v
                    pass
                else:
                    internal_edges.append(e)
            
            # Add mixed cliques
            for u, v in cross_edges:
                cliques.append([u, v])
                
            # Add pure C clique
            if k >= 2:
                cliques.append(list(range(k)))
                
            return cliques

        results = []
        for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
            results.append(instance(*spec))
            
        return results
