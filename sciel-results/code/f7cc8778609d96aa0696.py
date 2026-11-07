import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        cliques = []
        
        # available_edges is a set of frozensets or tuples (min, max)
        available_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                available_edges.add((u, w))
                
        def remove_edges(S):
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    e = (min(a, b), max(a, b))
                    if e in available_edges:
                        available_edges.remove(e)

        # Sort I vertices by neighborhood size (largest first)
        vs = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            N_v = neighbors[v]
            if not N_v:
                continue
            
            # Greedily build S_v
            S_v = []
            candidates = list(N_v)
            
            while candidates:
                # Filter candidates to those compatible with all in S_v
                valid_candidates = []
                for u in candidates:
                    compatible = True
                    for x in S_v:
                        e = (min(u, x), max(u, x))
                        if e not in available_edges:
                            compatible = False
                            break
                    if compatible:
                        valid_candidates.append(u)
                        
                if not valid_candidates:
                    break
                
                # Pick the best candidate
                # Heuristic: max degree in the subgraph of valid_candidates using available_edges
                best_u = -1
                best_deg = -1
                for u in valid_candidates:
                    deg = 0
                    for w in valid_candidates:
                        if u == w: continue
                        e = (min(u, w), max(u, w))
                        if e in available_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
                        
                S_v.append(best_u)
                candidates.remove(best_u)
                
            if S_v:
                cliques.append([v] + S_v)
                remove_edges(S_v)
        
        # Cover remaining edges in available_edges
        # Greedy clique cover
        while available_edges:
            edge = next(iter(available_edges))
            u, w = edge
            
            # Find maximal clique containing u and w in available_edges
            # Start with {u, w}
            clique = {u, w}
            # Try to add other vertices
            # A vertex x can be added if (u, x) and (w, x) are in available_edges
            # AND (x, y) is in available_edges for all y already in clique?
            # Actually, we are building a clique in the graph defined by available_edges.
            # So we need all pairs in the clique to be in available_edges.
            
            # Greedy expansion:
            changed = True
            while changed:
                changed = False
                for x in range(k):
                    if x in clique:
                        continue
                    # Check if x can be added
                    can_add = True
                    for y in clique:
                        e = (min(x, y), max(x, y))
                        if e not in available_edges:
                            can_add = False
                            break
                    if can_add:
                        clique.add(x)
                        changed = True
            
            # Add this clique
            cliques.append(list(clique))
            
            # Remove all edges within this clique from available_edges
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    e = (min(a, b), max(a, b))
                    if e in available_edges:
                        available_edges.remove(e)
                        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback
            pass
        cliques = solve_instance(*spec)
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
