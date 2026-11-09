# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _random_graph_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch_maximal_cliques(n, adj):
    """Bron-Kerbosch with pivot to enumerate all maximal cliques."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X to minimize |P \ N(u)|
        union = P | X
        if not union:
            return
        # Pick pivot with max degree in P
        pivot = max(union, key=lambda u: len(adj[u] & P))
        candidates = P - adj[pivot]
        
        for v in list(candidates):
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R_new, P_new, X_new)
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _greedy_hitting_set(cliques, n):
    """Greedy: repeatedly pick vertex covering most uncovered cliques."""
    if not cliques:
        return set()
    
    # Map each vertex to set of clique indices it belongs to
    vertex_cliques = {v: set() for v in range(n)}
    for i, c in enumerate(cliques):
        for v in c:
            vertex_cliques[v].add(i)
    
    uncovered = set(range(len(cliques)))
    selected = set()
    
    while uncovered:
        best_v = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = len(vertex_cliques[v] & uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count <= 0:
            break
        selected.add(best_v)
        uncovered -= vertex_cliques[best_v]
    
    return selected

def _local_search_hitting_set(cliques, n, initial_set, time_limit=10.0):
    """Local search with removal, 2-for-1, and swap improvements."""
    if not cliques:
        return set()
    
    selected = set(initial_set)
    
    # Precompute: for each clique, which vertices are in it
    clique_vertices = [set(c) for c in cliques]
    
    # For each vertex, which cliques does it cover
    vertex_cliques = {v: set() for v in range(n)}
    for i, c in enumerate(clique_vertices):
        for v in c:
            vertex_cliques[v].add(i)
    
    def is_valid(s):
        for i, c in enumerate(clique_vertices):
            if not (c & s):
                return False
        return True
    
    def uncovered_cliques(s):
        unc = []
        for i, c in enumerate(clique_vertices):
            if not (c & s):
                unc.append(i)
        return unc
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Move 1: try removing each selected vertex
        for v in list(selected):
            if (time.time() - start_time) > time_limit:
                break
            candidate = selected - {v}
            unc = uncovered_cliques(candidate)
            if not unc:
                selected = candidate
                improved = True
                break
            # 1-for-1: remove v, add w to cover unc (net same size, but may enable further removals)
            if len(unc) <= 3:
                unc_set = set(unc)
                # Find w that covers all of unc
                for w in range(n):
                    if w in selected:
                        continue
                    if vertex_cliques[w] >= unc_set:
                        new_set = candidate | {w}
                        if len(new_set) == len(selected):
                            # Same size but different composition; accept if it allows future removal
                            # Only accept if w has more total coverage than v
                            if len(vertex_cliques[w]) > len(vertex_cliques[v]):
                                selected = new_set
                                improved = True
                                break
                if improved:
                    break
        
        if improved:
            continue
        
        # Move 2: try removing two vertices (2-for-0)
        sel_list = list(selected)
        for i in range(len(sel_list)):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, len(sel_list)):
                v1, v2 = sel_list[i], sel_list[j]
                candidate = selected - {v1, v2}
                unc = uncovered_cliques(candidate)
                if not unc:
                    selected = candidate
                    improved = True
                    break
                # 2-for-1: remove two, add one
                if unc:
                    unc_set = set(unc)
                    best_add = None
                    best_cover = -1
                    for w in range(n):
                        if w in candidate:
                            continue
                        cover = len(vertex_cliques[w] & unc_set)
                        if cover > best_cover:
                            best_cover = cover
                            best_add = w
                    if best_add is not None and best_cover == len(unc_set):
                        new_set = candidate | {best_add}
                        if len(new_set) < len(selected):
                            selected = new_set
                            improved = True
                            break
            if improved:
                break
        
        if improved:
            continue
        
        # Move 3: try removing three vertices, adding one (3-for-1)
        sel_list = list(selected)
        for i in range(len(sel_list)):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, len(sel_list)):
                if (time.time() - start_time) > time_limit:
                    break
                for k in range(j + 1, len(sel_list)):
                    v1, v2, v3 = sel_list[i], sel_list[j], sel_list[k]
                    candidate = selected - {v1, v2, v3}
                    unc = uncovered_cliques(candidate)
                    if not unc:
                        selected = candidate
                        improved = True
                        break
                    if unc:
                        unc_set = set(unc)
                        best_add = None
                        best_cover = -1
                        for w in range(n):
                            if w in candidate:
                                continue
                            cover = len(vertex_cliques[w] & unc_set)
                            if cover > best_cover:
                                best_cover = cover
                                best_add = w
                        if best_add is not None and best_cover == len(unc_set):
                            new_set = candidate | {best_add}
                            if len(new_set) < len(selected):
                                selected = new_set
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return selected

def construct():
    deadline = time.time() + 130.0  # safety margin
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = _random_graph_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        cliques = _bron_kerbosch_maximal_cliques(n, adj)
        
        # Filter to cliques of size >= 2 (already done in BK)
        # Greedy initial solution
        greedy_set = _greedy_hitting_set(cliques, n)
        
        # Local search with time budget
        remaining = max(1.0, deadline - time.time() - 5.0)  # leave 5s safety margin
        per_graph_time = min(remaining / (3 - len(results)), 50.0)
        
        final_set = _local_search_hitting_set(cliques, n, greedy_set, time_limit=per_graph_time)
        
        results.append(sorted(final_set))
        
        # Check deadline
        if time.time() > deadline:
            # Return what we have, pad if needed
            break
    
    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])
    
    return results
# EVOLVE-BLOCK-END
