"""Clique partition of dense random graphs – greedy maximal-clique + consistent edge assignment."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    edges = set()
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
    return n, edges, adj

def _try_partition(n, edge_set, adj, deadline):
    """Find a valid clique partition. Returns list of cliques or None."""
    # Step 1: Find candidate cliques (greedy maximal cliques)
    candidate_cliques = []
    seen = set()
    
    # Greedy clique expansion from each vertex
    for v in range(n):
        if time.time() > deadline:
            break
        clique = {v}
        candidates = set(adj[v])
        while candidates:
            best = None
            best_score = -1
            for w in candidates:
                score = len(clique & adj[w])
                if score > best_score:
                    best_score = score
                    best = w
            if best is None or best_score < len(clique):
                break
            clique.add(best)
            candidates = {w for w in candidates if w != best and all(w in adj[c] for c in clique if c != w)}
        if len(clique) >= 2:
            key = frozenset(clique)
            if key not in seen:
                seen.add(key)
                candidate_cliques.append(key)
    
    # Step 2: Select a subset of cliques such that every edge is covered exactly once
    # Greedy: process cliques by size desc, include if all its edges are not yet covered
    candidate_cliques.sort(key=len, reverse=True)
    
    covered_edges = set()
    selected_cliques = []
    
    for c in candidate_cliques:
        if time.time() > deadline:
            break
        verts = sorted(c)
        edges_in_c = set()
        for a, b in combinations(verts, 2):
            e = (min(a, b), max(a, b))
            if e in edge_set:
                edges_in_c.add(e)
        # Only include if none of its edges are already covered
        if edges_in_c and not (edges_in_c & covered_edges):
            selected_cliques.append(c)
            covered_edges |= edges_in_c
    
    # Step 3: For uncovered edges, create 2-cliques
    for e in edge_set:
        if e not in covered_edges:
            selected_cliques.append(frozenset([e[0], e[1]]))
            covered_edges.add(e)
    
    # Validate
    for c in selected_cliques:
        verts = sorted(c)
        for a, b in combinations(verts, 2):
            if (min(a, b), max(a, b)) not in edge_set:
                return None  # Not a valid clique
    
    # Check every edge is in exactly one clique
    edge_count = {}
    for c in selected_cliques:
        verts = sorted(c)
        for a, b in combinations(verts, 2):
            e = (min(a, b), max(a, b))
            if e in edge_set:
                edge_count[e] = edge_count.get(e, 0) + 1
    for e in edge_set:
        if edge_count.get(e, 0) != 1:
            return None
    
    return [sorted(c) for c in selected_cliques]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            n, edge_set, adj = _generate_graph(n, p, seed)
            results.append([list(e) for e in sorted(edge_set)])
            continue
        
        n, edge_set, adj = _generate_graph(n, p, seed)
        
        # Try with different random seeds for tie-breaking
        best = None
        for attempt in range(5):
            if time.time() > deadline:
                break
            # Shuffle candidate clique order for diversity
            random.Random(100 + attempt).shuffle  # just for potential future use
            result = _try_partition(n, edge_set, adj, deadline)
            if result is not None:
                if best is None or len(result) < len(best):
                    best = result
        
        if best is None:
            # Fallback
            best = [list(e) for e in sorted(edge_set)]
        
        results.append(best)
    
    return results
# EVOLVE-BLOCK-END
