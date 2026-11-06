import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def complement_adj(n, adj):
        comp_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v and v not in adj[u]:
                    comp_adj[u].add(v)
        return comp_adj

    def max_clique(adj, n):
        # Tomita's algorithm with greedy coloring
        best_clique = set()
        
        # Greedy coloring of a set of vertices P
        def color_sort(P, adj):
            # Returns list of (vertex, color) sorted by color descending
            # Colors are assigned greedily
            uncolored = list(P)
            colors = {}
            # Simple greedy: assign lowest possible color
            # To be effective, we should order vertices, e.g., by degree in P
            # But for n=40, simple iteration is fine.
            # We need the color of each vertex to be an upper bound.
            # Standard Tomita: 
            # 1. Order vertices in P
            # 2. Assign colors such that adjacent vertices have different colors
            # 3. The color of v is the number of the color class it belongs to.
            
            # A simpler bound: just use the size of the largest independent set in P? No, that's hard.
            # Greedy coloring:
            # Initialize color[v] = 0 for all v in P
            # Iterate through vertices in some order
            # Assign the smallest color not used by neighbors
            
            # Actually, we just need the *number* of colors used, and the color of each vertex.
            # Let's use a standard greedy coloring.
            
            # To make the bound tight, we should process vertices in a way that 
            # high-degree vertices get lower colors? 
            # Tomita's approach:
            # Sort P by degree in P (descending).
            # Assign colors sequentially.
            
            ordered_P = sorted(P, key=lambda x: len(adj[x] & P), reverse=True)
            color_map = {}
            color_classes = []
            for v in ordered_P:
                used_colors = set()
                for u in adj[v]:
                    if u in color_map:
                        used_colors.add(color_map[u])
                # Find smallest available color
                c = 0
                while c in used_colors:
                    c += 1
                color_map[v] = c
                if c >= len(color_classes):
                    color_classes.append(set())
                color_classes[c].add(v)
            
            # Sort vertices by color descending
            # Vertices with higher color can potentially form larger cliques
            sorted_vertices = sorted(P, key=lambda x: color_map[x], reverse=True)
            return sorted_vertices, color_map

        def expand(R, P, X):
            nonlocal best_clique
            
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return
            
            if len(R) + len(P) <= len(best_clique):
                return
            
            # Color P
            sorted_P, color_map = color_sort(P, adj)
            
            # Process in reverse order of color
            # We need to maintain P and X as we go
            # Standard Tomita implementation details:
            # We iterate through sorted_P.
            # For each v, we check if |R| + color_map[v] > len(best_clique)
            # If yes, we recurse with R U {v}, P ∩ N(v), X ∩ N(v)
            # Then we remove v from P and add to X
            
            # To do this efficiently, we often modify P and X in place or use copies.
            # Given n=40, copies are fine.
            
            P_set = set(P)
            X_set = set(X)
            
            for v in sorted_P:
                if v not in P_set:
                    continue
                
                # Pruning check
                if len(R) + color_map[v] <= len(best_clique):
                    # Since sorted by color descending, if this fails, 
                    # all subsequent vertices (with lower or equal color) will also fail?
                    # Yes, if color_map is non-increasing in the sorted list.
                    # Our sort is reverse=True, so colors are non-increasing.
                    # So we can break.
                    break
                
                # Recurse
                new_R = R | {v}
                new_P = P_set & adj[v]
                new_X = X_set & adj[v]
                
                expand(new_R, new_P, new_X)
                
                # Update P and X
                P_set.remove(v)
                X_set.add(v)
                
                # Optional: update best_clique if R was a clique (it is)
                # But expand handles it when P and X are empty.
                # Actually, we should check if new_R is a valid clique (it is by construction)
                # and if it's larger than best.
                # But the base case in expand handles it.
                
        # Initial call
        # R is empty, P is all vertices, X is empty
        all_vertices = set(range(n))
        expand(set(), all_vertices, set())
        
        return best_clique

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Fallback: return a valid but potentially suboptimal set
            # e.g., all vertices
            for remaining_n, remaining_p, remaining_seed in instances[len(results):]:
                results.append(list(range(remaining_n)))
            return results

        edges = random_graph(n, p, seed)
        adj = build_adj(n, edges)
        comp_adj = complement_adj(n, adj)
        
        # Find Max Clique in Comp Graph -> Max Independent Set in Original Graph
        max_clique_set = max_clique(comp_adj, n)
        
        # Transversal is V \ I
        transversal = [v for v in range(n) if v not in max_clique_set]
        
        results.append(sorted(transversal))
        
    return results
