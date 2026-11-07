"""Clique partition of split graphs via greedy absorption + Max-Clique cover."""
import time
from random import Random


def construct():
    _T0 = time.time()
    _DEADLINE = 135.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _cc_edge(a, b):
        return (a, b) if a < b else (b, a)

    def _max_clique(adj, n):
        """Find a maximum clique in a graph with n vertices and adjacency list adj.
        Uses Bron-Kerbosch with pivot.
        Returns the clique as a set of vertices.
        """
        best_clique = set()
        
        def bronk(R, P, X):
            nonlocal best_clique
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return
            
            # Choose pivot u from P U X
            pivot = None
            max_deg = -1
            for u in P:
                deg = len(adj[u] & P)
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            if not P:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return

            # Iterate over P \ N(pivot)
            candidates = list(P - adj[pivot]) if pivot is not None else list(P)
            for v in candidates:
                R.add(v)
                bronk(R, P & adj[v], X & adj[v])
                R.discard(v)
                P.discard(v)
                X.add(v)
                
                # Prune if can't beat best
                if len(R) + len(P) <= len(best_clique):
                    break

        bronk(set(), set(range(n)), set())
        return best_clique

    def _cover_remaining_cc(remaining, k):
        """Cover the remaining C-C edges with C-only cliques.
        Uses Maximum Clique extraction.
        """
        if not remaining:
            return []
        
        cliques = []
        remaining = set(remaining)
        
        while remaining:
            # Build adjacency for current remaining edges
            adj = [set() for _ in range(k)]
            active = set()
            for (a, b) in remaining:
                adj[a].add(b)
                adj[b].add(a)
                active.add(a)
                active.add(b)
            
            if not active:
                break

            active_list = list(active)
            idx_map = {v: i for i, v in enumerate(active_list)}
            m = len(active_list)
            local_adj = [set() for _ in range(m)]
            
            for (a, b) in remaining:
                if a in idx_map and b in idx_map:
                    ia = idx_map[a]
                    ib = idx_map[b]
                    local_adj[ia].add(ib)
                    local_adj[ib].add(ia)
            
            max_cl = _max_clique(local_adj, m)
            
            if not max_cl:
                break
            
            clique_vertices = {active_list[i] for i in max_cl}
            
            cl_sorted = sorted(clique_vertices)
            for i in range(len(cl_sorted)):
                for j in range(i + 1, len(cl_sorted)):
                    remaining.discard(_cc_edge(cl_sorted[i], cl_sorted[j]))
            
            cliques.append(cl_sorted)
            
        return cliques

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add(_cc_edge(a, b))

        # N_C[v] = neighbors of I-vertex v in C
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if _cc_edge(u, v) in edge_set)

        # all C-C edges
        all_cc = set(_cc_edge(u, v) for u in range(k) for v in range(u + 1, k))
        cc_list = list(all_cc)

        # For each C-C edge (u,v), find I-vertices adjacent to both u and v
        cc_to_iverts = {}
        for idx, (u, v) in enumerate(cc_list):
            candidates = []
            for w in range(k, n):
                if u in N_C[w] and v in N_C[w]:
                    candidates.append(w)
            cc_to_iverts[idx] = candidates

        # Greedy: cover C-C edges with Type-B cliques (triangles)
        # Process C-C edges in order of most constrained first (fewest candidates)
        used_cc = set()
        covered_cross = set()
        cliques = []
        
        cc_order = sorted(range(len(cc_list)), key=lambda i: len(cc_to_iverts[i]))
        
        for idx in cc_order:
            u, v = cc_list[idx]
            if (u, v) in used_cc:
                continue
            for w in cc_to_iverts[idx]:
                if _cc_edge(u, w) not in covered_cross and _cc_edge(v, w) not in covered_cross:
                    cliques.append([u, v, w])
                    used_cc.add((u, v))
                    covered_cross.add(_cc_edge(u, w))
                    covered_cross.add(_cc_edge(v, w))
                    break
        
        # Cover remaining cross-edges with 2-cliques
        for w in range(k, n):
            for u in sorted(N_C[w]):
                if _cc_edge(u, w) not in covered_cross:
                    cliques.append([u, w])
                    covered_cross.add(_cc_edge(u, w))
        
        # Cover remaining C-C edges with C-only cliques
        remaining_cc = all_cc - used_cc
        c_only = _cover_remaining_cc(remaining_cc, k)
        for cl in c_only:
            cliques.append(cl)
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue
        results.append(solve_one(*spec))

    # convert to plain python int lists
    out = []
    for clist in results:
        nl = []
        for c in clist:
            nl.append([int(x) for x in c])
        out.append(nl)
    return out
