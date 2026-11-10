import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(edges, n):
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: add remaining edges as pairs
                for e in remaining_edges:
                    cliques.append(list(e))
                break
                
            # Pick best edge
            best_edge = None
            best_score = -1
            for e in remaining_edges:
                u, v = e
                score = len(adj[u]) + len(adj[v])
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                break
                
            u, v = best_edge
            clique = [u, v]
            clique_set = {u, v}
            
            def is_edge_present(a, b):
                if a > b: a, b = b, a
                return (a, b) in remaining_edges

            # Initial candidates
            candidates = adj[u].intersection(adj[v])
            valid_candidates = set()
            for w in candidates:
                if is_edge_present(u, w) and is_edge_present(v, w):
                    valid_candidates.add(w)
            
            while valid_candidates:
                # Pick best candidate
                best_w = None
                best_deg = -1
                for w in valid_candidates:
                    deg = len(adj[w])
                    if deg > best_deg:
                        best_deg = deg
                        best_w = w
                
                if best_w is None:
                    break

                w = best_w
                clique.append(w)
                clique_set.add(w)
                
                new_valid_candidates = set()
                for x in valid_candidates:
                    if x == w:
                        continue
                    if is_edge_present(w, x):
                        new_valid_candidates.add(x)
                
                valid_candidates = new_valid_candidates

            cliques.append(clique)
            
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a > b: a, b = b, a
                    remaining_edges.discard((a, b))
            
        return cliques

    n1, p1, s1 = 32, 0.6, 601
    edges1 = random_graph(n1, p1, s1)
    result1 = greedy_partition(edges1, n1)
    
    n2, p2, s2 = 64, 0.55, 602
    edges2 = random_graph(n2, p2, s2)
    result2 = greedy_partition(edges2, n2)
    
    return [result1, result2]
