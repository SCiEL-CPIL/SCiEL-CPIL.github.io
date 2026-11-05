"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from random import Random

def _generate_graph(n, p, seed):
    """Generate graph edges matching the verifier's random_graph."""
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed, deadline):
    """Find a clique partition of the graph minimizing number of cliques."""
    edges = _generate_graph(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
    
    remaining_edges = set(edge_set)
    cliques = []
    
    # Precompute triangle count for each edge
    tri_count = {}
    for (u, v) in edge_set:
        cnt = 0
        for w in adj[u]:
            if w != v and (min(v, w), max(v, w)) in edge_set:
                cnt += 1
        tri_count[(u, v)] = cnt
    
    while remaining_edges:
        if time.time() > deadline:
            break
        
        # Pick edge with highest triangle count among remaining
        best_e = None
        best_tri = -1
        for e in remaining_edges:
            t = tri_count.get(e, 0)
            if t > best_tri:
                best_tri = t
                best_e = e
        
        if best_e is None:
            break
        
        u0, v0 = best_e
        
        # Build maximal clique containing (u0, v0) via greedy expansion
        clique = [u0, v0]
        cset = {u0, v0}
        
        while True:
            if time.time() > deadline:
                break
            # Find all vertices adjacent to all clique members via remaining edges
            cands = None
            for c in cset:
                if cands is None:
                    cands = set(adj[c])
                else:
                    cands &= adj[c]
            valid_cands = []
            for w in cands:
                if w in cset:
                    continue
                ok = True
                for c in cset:
                    e = (min(w, c), max(w, c))
                    if e not in remaining_edges:
                        ok = False
                        break
                if ok:
                    valid_cands.append(w)
            
            if not valid_cands:
                break
            
            # Pick candidate with most connections to other valid candidates
            best_w = None
            best_score = -1
            for w in valid_cands:
                score = 0
                for other in valid_cands:
                    if other == w:
                        continue
                    e = (min(w, other), max(w, other))
                    if e in remaining_edges:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_w = w
            
            if best_w is None:
                break
            
            clique.append(best_w)
            cset.add(best_w)
        
        clique.sort()
        if len(clique) >= 2:
            cliques.append(clique)
            # Remove edges in clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
            # Update triangle counts for remaining edges
            for e in remaining_edges:
                eu, ev = e
                cnt = 0
                for w in adj[eu]:
                    if w != ev:
                        e1 = (min(eu, w), max(eu, w))
                        e2 = (min(ev, w), max(ev, w))
                        if e1 in remaining_edges and e2 in remaining_edges:
                            cnt += 1
                tri_count[e] = cnt
    
    # Handle any remaining edges
    for e in remaining_edges:
        cliques.append([e[0], e[1]])
    
    return cliques

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            edges = _generate_graph(n, p, seed)
            result.append([[u, v] for u, v in edges])
        else:
            cliques = _solve_instance(n, p, seed, deadline)
            result.append(cliques)
    
    return result
# EVOLVE-BLOCK-END
