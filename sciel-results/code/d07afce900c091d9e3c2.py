import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

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
        
        # Pre-compute for each pair (u, w) in K, the list of v in I connected to both
        common_neighbors = {}
        for u in range(k):
            for w in range(u + 1, k):
                common_neighbors[(u, w)] = []
        
        for v in range(k, n):
            N_v = neighbors[v]
            for i in range(len(N_v)):
                for j in range(i + 1, len(N_v)):
                    u, w = N_v[i], N_v[j]
                    pair = (min(u, w), max(u, w))
                    common_neighbors[pair].append(v)
        
        cliques = []
        
        # Step 1: Maximize triangles with improved greedy
        available_k_edges = set((u, w) for u in range(k) for w in range(u + 1, k))
        k_edges_sorted = sorted(available_k_edges, key=lambda e: len(common_neighbors[e]), reverse=True)
        used_cross_edges = set()
        
        for (u, w) in k_edges_sorted:
            if (u, w) not in available_k_edges:
                continue
            candidates = common_neighbors[(u, w)]
            best_v = None
            best_score = -1
            for v in candidates:
                e1 = (u, v)
                e2 = (w, v)
                if e1 not in used_cross_edges and e2 not in used_cross_edges:
                    # Score: prefer v with fewer other available cross-edges
                    # (i.e., v is "scarce" so using it now is less wasteful)
                    other_edges = 0
                    for u2 in neighbors[v]:
                        if u2 != u and u2 != w:
                            e = (u2, v)
                            if e not in used_cross_edges:
                                other_edges += 1
                    score = -other_edges  # lower is better
                    if score > best_score:
                        best_score = score
                        best_v = v
            if best_v is not None:
                cliques.append([u, w, best_v])
                used_cross_edges.add((u, best_v))
                used_cross_edges.add((w, best_v))
                available_k_edges.remove((u, w))
        
        # Step 2: Cover remaining cross-edges with 2-cliques
        for v in range(k, n):
            N_v = neighbors[v]
            for u in N_v:
                e = (u, v)
                if e not in used_cross_edges:
                    cliques.append([u, v])
                    used_cross_edges.add(e)
        
        # Step 3: Cover remaining K-edges
        remaining_k_edges = list(available_k_edges)
        
        while remaining_k_edges:
            if time.time() > deadline:
                for e in remaining_k_edges:
                    cliques.append([e[0], e[1]])
                break
            
            u, w = remaining_k_edges[0]
            
            candidates = set()
            for e in remaining_k_edges:
                a, b = e
                if a == u and b != w:
                    candidates.add(b)
                elif b == u and a != w:
                    candidates.add(a)
                if a == w and b != u:
                    candidates.add(b)
                elif b == w and a != u:
                    candidates.add(a)
            
            valid_c = []
            for c in candidates:
                e_uc = (min(u, c), max(u, c))
                e_wc = (min(w, c), max(w, c))
                if e_uc in remaining_k_edges and e_wc in remaining_k_edges:
                    valid_c.append(c)
            
            if valid_c:
                c = valid_c[0]
                clique = [u, w, c]
                edges_to_remove = set()
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        edges_to_remove.add((min(a, b), max(a, b)))
                remaining_k_edges = [e for e in remaining_k_edges if e not in edges_to_remove]
                cliques.append(clique)
            else:
                cliques.append([u, w])
                remaining_k_edges.remove((u, w))
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            pass
        cliques = solve_instance(*spec)
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
