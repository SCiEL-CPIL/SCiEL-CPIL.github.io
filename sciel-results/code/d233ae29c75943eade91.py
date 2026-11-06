import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def find_maximal_cliques(n, edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot
            u = None
            max_inter = -1
            for v in P.union(X):
                inter = len(P.intersection(adj[v]))
                if inter > max_inter:
                    max_inter = inter
                    u = v
            
            candidates = P.difference(adj[u]) if u is not None else P
            for v in candidates:
                bronk(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)
                
        bronk(frozenset(), frozenset(range(n)), frozenset())
        return cliques

    def max_clique_in_subgraph(verts, adj_map):
        best = []
        
        def dfs(current, candidates):
            nonlocal best
            if len(current) > len(best):
                best = list(current)
            if not candidates:
                return
            # Simple heuristic: order by degree in candidates
            # sorted_cands = sorted(candidates, key=lambda x: len(candidates.intersection(adj_map[x])), reverse=True)
            # for v in sorted_cands:
            for v in list(candidates):
                new_current = current.union({v})
                new_candidates = candidates.intersection(adj_map[v])
                dfs(new_current, new_candidates)
                candidates.remove(v)
                
        dfs(frozenset(), frozenset(verts))
        return best

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        cliques = find_maximal_cliques(n, edges)
        
        # Precompute edge sets for cliques
        cliques_with_edges = []
        for c in cliques:
            c_edges = set()
            verts = list(c)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u, v = verts[i], verts[j]
                    if u > v: u, v = v, u
                    c_edges.add((u, v))
            cliques_with_edges.append((c, c_edges))
            
        # Sort by number of edges descending
        cliques_with_edges.sort(key=lambda x: len(x[1]), reverse=True)
        
        uncovered = set(edges)
        solution = []
        
        # Phase 1: Greedy maximal cliques
        for c, c_edges in cliques_with_edges:
            if time.time() > deadline:
                break
            if c_edges.issubset(uncovered):
                solution.append(list(c))
                uncovered -= c_edges
                
        # Phase 2: Fill in remaining
        while uncovered and time.time() < deadline:
            # Find max clique in subgraph induced by uncovered
            verts_in_uncovered = set()
            for u, v in uncovered:
                verts_in_uncovered.add(u)
                verts_in_uncovered.add(v)
            
            sub_adj = defaultdict(set)
            for u, v in uncovered:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
                
            max_c = max_clique_in_subgraph(verts_in_uncovered, sub_adj)
            
            if len(max_c) >= 2:
                solution.append(list(max_c))
                # Remove edges in max_c
                c_edges = set()
                verts = list(max_c)
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        u, v = verts[i], verts[j]
                        if u > v: u, v = v, u
                        c_edges.add((u, v))
                uncovered -= c_edges
            else:
                # Fallback: single edge
                e = next(iter(uncovered))
                solution.append(list(e))
                uncovered.remove(e)
                
        # If time ran out, add remaining edges as single cliques
        if uncovered:
            for e in uncovered:
                solution.append(list(e))
                
        return solution

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            edges = random_graph(n, p, seed)
            results.append([list(e) for e in edges])
        else:
            results.append(solve_instance(n, p, seed))
            
    return results
