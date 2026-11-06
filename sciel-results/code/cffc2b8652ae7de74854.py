def construct():
        import numpy as np
        from random import Random

        def instance_edges(n, p, seed):
            rng = Random(seed)
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        edges.append((u, v))
            return edges

        def get_complement_adj(n, edges):
            # Build adjacency list for complement graph
            # Original edges are in 'edges'
            # Complement edges are all pairs NOT in 'edges'
            adj = [set() for _ in range(n)]
            edge_set = set()
            for u, v in edges:
                edge_set.add((u, v))
                edge_set.add((v, u))
            
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in edge_set:
                        adj[u].add(v)
                        adj[v].add(u)
            return adj

        def max_clique_size_and_vertices(adj, n):
            # Bron-Kerbosch with Pivot and Branch and Bound
            # We want to find the maximum clique in the graph defined by adj
            best_clique = []
            best_size = 0
            
            # Pre-compute vertex ordering for better pruning (optional but helpful)
            # Simple heuristic: order by degree descending
            # But for n=40, simple B&B is fine.
            
            def expand(R, P, X):
                nonlocal best_clique, best_size
                if not P and not X:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = list(R)
                    return
                
                if P:
                    # Pivot selection: choose u in P ∪ X that maximizes |P ∩ N(u)|
                    u = max(P | X, key=lambda x: len(P & adj[x]))
                    candidates = P - adj[u]
                else:
                    candidates = P
                    
                for v in list(candidates):
                    # Pruning: even if we add all remaining vertices in P, can we beat best_size?
                    if len(R) + len(P) <= best_size:
                        return
                    # Note: The pruning above is a bit loose. 
                    # Better: if len(R) + len(candidates) <= best_size, we might prune branches, 
                    # but standard B&B checks at the start of expand or before recursion.
                    
                    expand(R | {v}, P & adj[v], X & adj[v])
                    P.remove(v)
                    X.add(v)
                    
            expand(set(), set(range(n)), set())
            return best_clique, best_size

        def solve_min_transversal(n, edges):
            # Transversal S hits all cliques size >= 2.
            # This is equivalent to V\S being an Independent Set in G.
            # So we want Max Independent Set in G.
            # Max Independent Set in G is Max Clique in Complement G.
            
            comp_adj = get_complement_adj(n, edges)
            mis_vertices, _ = max_clique_size_and_vertices(comp_adj, n)
            
            # The vertices in mis_vertices form the Max Independent Set.
            # The transversal is the rest.
            mis_set = set(mis_vertices)
            transversal = [v for v in range(n) if v not in mis_set]
            return sorted(transversal)

        out = []
        for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
            edges = instance_edges(n, p, seed)
            sol = solve_min_transversal(n, edges)
            out.append(sol)
            
        return out
