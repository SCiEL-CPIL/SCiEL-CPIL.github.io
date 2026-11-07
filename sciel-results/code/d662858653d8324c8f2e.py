import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique_bk(verts, adj, time_limit=None):
        """Bron-Kerbosch with pivot to find maximum clique."""
        best = []
        
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            U = P | X
            if U:
                pivot = max(U, key=lambda u: len(P & adj[u]))
                candidates = list(P - adj[pivot])
            else:
                candidates = list(P)
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(verts), set())
        return best

    def clique_partition(n, k, p, seed):
        rng = random.Random(seed)
        adj_I = [set() for _ in range(n - k)]
        
        for i in range(n - k):
            v = k + i
            for u in range(k):
                if rng.random() < p:
                    adj_I[i].add(u)
        
        used_K_edges = set()
        cliques = []
        
        i_indices = list(range(n - k))
        rng.shuffle(i_indices)
        
        for i in i_indices:
            v = k + i
            U = set(adj_I[i])
            while U:
                # Build available graph on U: edge (u,w) if (u,w) NOT in used_K_edges
                # Find max clique in available graph
                avail_adj = {u: set() for u in U}
                for u in U:
                    for w in U:
                        if u == w:
                            continue
                        if (min(u, w), max(u, w)) not in used_K_edges:
                            avail_adj[u].add(w)
                
                # Convert to list-indexed for BK
                verts_list = sorted(U)
                vert_to_idx = {v: i for i, v in enumerate(verts_list)}
                adj_list = [set() for _ in range(len(verts_list))]
                for u in U:
                    for w in avail_adj[u]:
                        adj_list[vert_to_idx[u]].add(vert_to_idx[w])
                
                C_idx = max_clique_bk(set(range(len(verts_list))), adj_list)
                C = set(verts_list[i] for i in C_idx)
                
                if not C:
                    u = next(iter(U))
                    C = {u}
                
                clique = [v] + sorted(C)
                cliques.append(clique)
                for u, w in combinations(sorted(C), 2):
                    used_K_edges.add((min(u, w), max(u, w)))
                U -= C
        
        # Cover remaining K edges
        rem_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in used_K_edges:
                    rem_edges.append((u, w))
        
        if rem_edges:
            adj_rem = [set() for _ in range(k)]
            for u, w in rem_edges:
                adj_rem[u].add(w)
                adj_rem[w].add(u)
            
            remaining = set(rem_edges)
            
            while remaining:
                verts = set()
                for u, w in remaining:
                    verts.add(u)
                    verts.add(w)
                
                C = max_clique_bk(verts, adj_rem)
                
                if not C:
                    break
                
                clique = sorted(C)
                cliques.append(clique)
                
                for u, w in combinations(sorted(C), 2):
                    edge = (min(u, w), max(u, w))
                    if edge in remaining:
                        remaining.remove(edge)
                
                for u in C:
                    for w in C:
                        if u != w:
                            if w in adj_rem[u]:
                                adj_rem[u].remove(w)
                                adj_rem[w].remove(u)

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
