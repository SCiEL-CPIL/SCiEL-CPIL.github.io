import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        N = {v: set() for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N[v].add(u)
        return N, k, n

    def get_cliques_for_instance(N, k, n):
        cliques = []
        covered_edges_in_K = set()
        
        for v in range(k, n):
            neighbors = N[v]
            if neighbors:
                clique = [v] + list(neighbors)
                if len(clique) >= 2:
                    cliques.append(clique)
                for u1, u2 in combinations(neighbors, 2):
                    covered_edges_in_K.add((min(u1, u2), max(u1, u2)))
            # If no neighbors, v is isolated in the context of edges to cover.
            # It doesn't contribute to covering K edges.
            # It doesn't have edges to cover itself.
            # So it is ignored.

        remaining_edges = []
        for u1 in range(k):
            for u2 in range(u1 + 1, k):
                if (u1, u2) not in covered_edges_in_K:
                    remaining_edges.append((u1, u2))
        
        if not remaining_edges:
            return cliques

        # Build adjacency for complement of remaining graph
        # Vertices 0..k-1
        # Edge in complement if NOT in remaining_edges
        comp_adj = [set() for _ in range(k)]
        rem_adj = [set() for _ in range(k)]
        for u1, u2 in remaining_edges:
            rem_adj[u1].add(u2)
            rem_adj[u2].add(u1)
        
        for u1 in range(k):
            for u2 in range(u1 + 1, k):
                if u2 not in rem_adj[u1]:
                    comp_adj[u1].add(u2)
                    comp_adj[u2].add(u1)
        
        # Greedy coloring of comp_adj with multiple orderings
        best_cliques = []
        best_size = float('inf')
        
        # Define orderings
        # 1. Natural
        orders = [list(range(k))]
        # 2. Reverse
        orders.append(list(range(k-1, -1, -1)))
        # 3. Degree in comp_adj descending
        orders.append(sorted(range(k), key=lambda x: len(comp_adj[x]), reverse=True))
        # 4. Degree in comp_adj ascending
        orders.append(sorted(range(k), key=lambda x: len(comp_adj[x])))
        # 5. Degree in rem_adj descending (high degree in rem = low in comp?)
        orders.append(sorted(range(k), key=lambda x: len(rem_adj[x]), reverse=True))
        # 6. Degree in rem_adj ascending
        orders.append(sorted(range(k), key=lambda x: len(rem_adj[x])))
        
        for order in orders:
            color_map = {}
            color_classes = {}
            
            for v in order:
                used_colors = set()
                for u in comp_adj[v]:
                    if u in color_map:
                        used_colors.add(color_map[u])
                
                c = 0
                while c in used_colors:
                    c += 1
                
                color_map[v] = c
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(v)
            
            current_cliques = []
            for c, verts in color_classes.items():
                if len(verts) >= 2:
                    current_cliques.append(sorted(verts))
            
            if len(current_cliques) < best_size:
                best_size = len(current_cliques)
                best_cliques = current_cliques
            
            if best_size == 1:
                break

        cliques.extend(best_cliques)
        
        final_cliques = []
        for c in cliques:
            c = list(set(c))
            if len(c) >= 2:
                c.sort()
                final_cliques.append(c)
        
        return final_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        n, k, p, seed = spec
        N, k_val, n_val = instance(n, k, p, seed)
        cliques = get_cliques_for_instance(N, k_val, n_val)
        results.append(cliques)
        
    return results
