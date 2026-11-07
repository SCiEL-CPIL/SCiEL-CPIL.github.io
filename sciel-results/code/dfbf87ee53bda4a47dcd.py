import time
from random import Random
from collections import deque

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _greedy_clique_cover_cc(k, edge_set, time_limit):
        """Cover C-C edges (among vertices 0..k-1) with minimal cliques.
        Since C is a complete graph, any subset is a clique.
        We need to partition all C-C edges into cliques.
        Use greedy: repeatedly find largest clique covering most uncovered edges."""
        if k <= 1:
            return []
        
        # All C-C edges
        all_cc = [(u, w) for u in range(k) for w in range(u + 1, k)]
        if not all_cc:
            return []
        
        uncovered = set(all_cc)
        cliques = []
        
        while uncovered:
            if time.time() > time_limit:
                break
            
            # Find the C-vertex with most uncovered incident edges
            best_v = None
            best_cnt = 0
            for v in range(k):
                cnt = sum(1 for e in uncovered if v in e)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            
            if best_v is None:
                break
            
            # Build maximal clique containing best_v that covers uncovered edges
            # Start with {best_v}, add vertices that form edges with all current members
            clique = [best_v]
            candidates = [v for v in range(k) if v != best_v]
            
            # Greedily add vertices
            for v in sorted(candidates, key=lambda x: -sum(1 for e in uncovered if x in e)):
                if time.time() > time_limit:
                    break
                # Check if v can be added: v must be adjacent to all in clique
                # (always true since C is complete)
                # But we want to maximize edge coverage
                # Add v if it covers at least one uncovered edge
                covers = any(v in e for e in uncovered)
                if covers:
                    clique.append(v)
            
            if len(clique) < 2:
                # Single vertex, can't form clique
                break
            
            # Remove covered edges
            clique_set = set(clique)
            for e in list(uncovered):
                if e[0] in clique_set and e[1] in clique_set:
                    uncovered.discard(e)
            
            cliques.append(sorted(clique))
        
        return cliques

    def _solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))
        
        # N_C[v] = set of C-vertices adjacent to I-vertex v
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)
        
        cross_edges_norm = []
        for (a, b) in edge_set:
            if a < k and b >= k:
                cross_edges_norm.append((a, b))
            elif b < k and a >= k:
                cross_edges_norm.append((b, a))
        
        # Strategy: For each I-vertex v with at least 2 C-neighbors, form clique {v} ∪ N_C[v]
        # This covers all cross edges incident to v and all C-C edges within N_C[v]
        # Then cover remaining C-C edges with C-only cliques
        
        # Option A: Use all I-vertices with |N_C[v]| >= 2
        # Option B: Select subset of I-vertices to minimize total cliques
        
        # Let's try: use all I-vertices that have cross edges
        # Each such I-vertex v contributes clique {v} ∪ N_C[v]
        
        cliques = []
        covered_cc = set()
        
        # First, add cliques for each I-vertex with cross edges
        for v in range(k, n):
            if N_C[v]:
                clique = sorted([v] + list(N_C[v]))
                if len(clique) >= 2:
                    cliques.append(clique)
                    # Mark C-C edges within N_C[v] as covered
                    nc = list(N_C[v])
                    for i in range(len(nc)):
                        for j in range(i + 1, len(nc)):
                            covered_cc.add((nc[i], nc[j]) if nc[i] < nc[j] else (nc[j], nc[i]))
        
        # Now find remaining C-C edges
        all_cc = [(u, w) for u in range(k) for w in range(u + 1, k)]
        remaining_cc = [e for e in all_cc if e not in covered_cc]
        
        if remaining_cc:
            # Cover remaining C-C edges with C-only cliques
            # Since C is complete, we can partition C-vertices into groups
            # Each group forms a clique covering all edges within it
            
            # Greedy: repeatedly pick the C-vertex with most remaining incident edges,
            # then build a maximal clique around it
            remaining_set = set(remaining_cc)
            c_cliques = []
            
            while remaining_set:
                if _time_up():
                    break
                
                # Find C-vertex with most remaining incident edges
                deg = [0] * k
                for (u, w) in remaining_set:
                    deg[u] += 1
                    deg[w] += 1
                
                best_v = max(range(k), key=lambda x: deg[x])
                if deg[best_v] == 0:
                    break
                
                # Build clique: start with best_v, add vertices connected to all in clique
                clique = [best_v]
                # Try adding vertices in order of degree
                candidates = sorted(range(k), key=lambda x: -deg[x])
                for v in candidates:
                    if v == best_v:
                        continue
                    # Check if v is adjacent to all in clique (always true in complete C)
                    # But we want to maximize edge coverage
                    # Add v if it covers at least one remaining edge
                    covers = any(v in e for e in remaining_set)
                    if covers:
                        clique.append(v)
                
                if len(clique) < 2:
                    break
                
                # Remove covered edges
                clique_set = set(clique)
                for e in list(remaining_set):
                    if e[0] in clique_set and e[1] in clique_set:
                        remaining_set.discard(e)
                
                c_cliques.append(sorted(clique))
            
            cliques.extend(c_cliques)
        
        # Verify validity
        if not cliques:
            # Fallback: individual edges
            for (a, b) in edge_set:
                cliques.append(sorted([a, b]))
            return cliques
        
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        # Invalid, fallback
                        cliques = []
                        for (a2, b2) in edge_set:
                            cliques.append(sorted([a2, b2]))
                        return cliques
        
        # Verify all edges covered exactly once
        edge_cover_count = {}
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    edge_cover_count[(a, b)] = edge_cover_count.get((a, b), 0) + 1
        
        for e in edge_set:
            if edge_cover_count.get(e, 0) != 1:
                # Invalid, fallback
                cliques = []
                for (a, b) in edge_set:
                    cliques.append(sorted([a, b]))
                return cliques
        
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
                    cliques.append(sorted([u, v]))
            results.append(cliques)
            continue
        n, k, p, seed = spec
        cliques = _solve_instance(n, k, p, seed)
        results.append(cliques)
    return results
