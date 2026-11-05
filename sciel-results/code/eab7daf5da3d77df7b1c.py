"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _random_graph(n, p, seed):
    """Generate edge list using same RNG as verifier."""
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adj(n, edges):
    """Build adjacency sets."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_clique(adj, start, n):
    """Grow a clique greedily from start vertex."""
    clique = [start]
    candidates = set(adj[start])
    while candidates:
        # Pick vertex with max degree within candidates
        best = max(candidates, key=lambda x: len(adj[x] & candidates))
        clique.append(best)
        candidates = candidates & adj[best]
    return clique

def _solve_instance(n, p, seed, deadline):
    """Solve one instance: partition edges into minimum cliques."""
    edges = _random_graph(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency
    adj = _build_adj(n, edges)
    
    # Track remaining edges
    remaining = set(edges)
    result = []
    
    # Strategy: repeatedly find large cliques
    # Use multiple rounds with different starting strategies
    while remaining:
        if time.time() > deadline - 5:
            # Fallback: add each remaining edge as a 2-clique
            for e in remaining:
                result.append(list(e))
            break
        
        # Find the best clique among several strategies
        best_clique = None
        best_size = 0
        
        # Strategy 1: Start from each vertex, grow greedily
        for v in range(n):
            if time.time() > deadline - 5:
                break
            # Only consider vertices that still have edges
            if not (adj[v] & {x for e in remaining for x in e if x != v}):
                continue
            clique = _greedy_clique(adj, v, n)
            # Check all edges in clique are in remaining
            valid = True
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    if e not in remaining:
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
        
        if best_clique is None or len(best_clique) < 2:
            # No clique of size >= 2 found; add single edges
            for e in remaining:
                result.append(list(e))
            break
        
        # Add clique and remove its edges
        result.append(best_clique)
        for i in range(len(best_clique)):
            for j in range(i + 1, len(best_clique)):
                e = (min(best_clique[i], best_clique[j]), max(best_clique[i], best_clique[j]))
                remaining.discard(e)
    
    return result

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        result = _solve_instance(n, p, seed, deadline)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
