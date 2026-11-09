import time
import itertools
import random
from collections import Counter, defaultdict

_START = time.time()
_LIMIT = 140.0  # Safety margin under 150s kill

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _edge_key(u, v):
    return (u, v) if u < v else (v, u)

def _adj_mask(n, edges):
    m = [0] * n
    for (u, v) in edges:
        m[u] |= 1 << v
        m[v] |= 1 << u
    return m

def _iter_bits(x):
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb

def _maximal_cliques_bit(adj_mask, n, min_size=3, cap=500000, deadline_ok=None):
    """Bron–Kerbosch with pivot, returning maximal cliques as bitmasks."""
    result = []
    aborted = [False]

    def bronk(R, P, X):
        if aborted[0]:
            return
        if deadline_ok is not None and not deadline_ok():
            aborted[0] = True
            return
        if not P and not X:
            if R.bit_count() >= min_size:
                result.append(R)
                if len(result) >= cap:
                    aborted[0] = True
            return
        if not P:
            return
        # Choose pivot
        ux = P | X
        best_u = -1
        best_c = -1
        for u in _iter_bits(ux):
            c = (P & adj_mask[u]).bit_count()
            if c > best_c:
                best_c = c
                best_u = u
        # Iterate over P \\ N(u)
        candidates = P & ~adj_mask[best_u] if best_u >= 0 else P
        for v in _iter_bits(candidates):
            bit = 1 << v
            bronk(R | bit, P & adj_mask[v], X & adj_mask[v])
            if aborted[0]:
                return
            P &= ~bit
            X |= bit

    bronk(0, (1 << n) - 1, 0)
    return result

def _clique_edges_mask(clique_mask):
    """Return a set of edge keys for a clique bitmask."""
    verts = list(_iter_bits(clique_mask))
    edges = set()
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            edges.add(_edge_key(verts[i], verts[j]))
    return edges

def _solve_instance(n, p, seed, max_rounds=30):
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    
    if not all_edges:
        return []

    adj_mask = _adj_mask(n, edges)
    
    # 1. Find all maximal cliques of size >= 3
    # We also need to handle edges that are not in any clique of size 3.
    # If an edge (u,v) is not in any triangle, it must be covered by a clique of size 2 (just itself).
    # So we should identify such edges early.
    
    maxcls = _maximal_cliques_bit(adj_mask, n, 3, 500000, _deadline_ok)
    
    # Map edges to cliques
    # For efficiency, we'll store cliques as lists of vertices and sets of edges.
    clique_data = []
    for mask in maxcls:
        verts = list(_iter_bits(mask))
        edge_set = _clique_edges_mask(mask)
        clique_data.append((verts, edge_set))
        
    # Identify edges not covered by any maximal clique of size >= 3
    covered_by_maximal = set()
    for _, edge_set in clique_data:
        covered_by_maximal.update(edge_set)
        
    isolated_edges = all_edges - covered_by_maximal
    
    # These isolated edges must be covered by single-edge cliques.
    # We can just add them directly to the result and remove from all_edges.
    chosen = []
    for e in isolated_edges:
        chosen.append([e[0], e[1]])
        
    remaining_edges = all_edges - isolated_edges
    
    if not remaining_edges:
        return chosen

    # 2. Greedy Set Cover on remaining_edges using maximal cliques
    # Weight of an edge: 1 / (number of cliques containing it)
    # We can approximate this with static weights from the full set of maximal cliques.
    
    edge_counts = Counter()
    for _, edge_set in clique_data:
        for e in edge_set:
            if e in remaining_edges:
                edge_counts[e] += 1
                
    # Initialize weights
    edge_weights = {e: 1.0 / count for e, count in edge_counts.items()}
    
    # For edges in remaining_edges but not in edge_counts (should not happen if maxcls covers all non-isolated),
    # assign high weight.
    for e in remaining_edges:
        if e not in edge_weights:
            edge_weights[e] = 1.0

    # We will pick cliques from clique_data.
    # To allow re-using cliques partially? No, a clique is a set of edges.
    # Once we pick a clique, we cover its edges.
    # We don't remove the clique from the pool, but we only count uncovered edges for gain.
    
    # Optimization: Sort cliques by initial potential gain?
    # Or just iterate.
    
    # To speed up, we can pre-compute the list of edges for each clique.
    # And maintain a set of remaining edges.
    
    # Let's use a list of active clique indices.
    # If a clique has no uncovered edges, it's useless.
    
    # Simple loop:
    while remaining_edges and _deadline_ok():
        best_gain = -1
        best_clique_idx = -1
        best_clique_edges_covered = set()
        
        for idx, (verts, edge_set) in enumerate(clique_data):
            # Calculate gain
            gain = 0.0
            covered_in_this_clique = set()
            for e in edge_set:
                if e in remaining_edges:
                    gain += edge_weights[e]
                    covered_in_this_clique.add(e)
            
            if gain > best_gain:
                best_gain = gain
                best_clique_idx = idx
                best_clique_edges_covered = covered_in_this_clique
                
        if best_clique_idx == -1 or best_gain <= 0:
            # No clique covers any remaining edges? Should not happen if maxcls is complete.
            # Fallback: pick any remaining edge as a single clique.
            e = next(iter(remaining_edges))
            chosen.append([e[0], e[1]])
            remaining_edges.remove(e)
            continue
            
        # Apply the chosen clique
        verts, _ = clique_data[best_clique_idx]
        chosen.append(verts)
        
        for e in best_clique_edges_covered:
            remaining_edges.remove(e)
            
        # Update weights? 
        # Static weights are faster. Dynamic would require updating all cliques containing removed edges.
        # Given the time limit, static weights are a good trade-off.
        
    # Add any leftover edges (should be empty)
    for e in remaining_edges:
        chosen.append([e[0], e[1]])
        
    return chosen

def construct():
    n1, p1, s1 = 32, 0.6, 601
    n2, p2, s2 = 64, 0.55, 602
    
    # Solve instance 1
    res1 = _solve_instance(n1, p1, s1)
    
    # Solve instance 2
    res2 = _solve_instance(n2, p2, s2)
    
    return [res1, res2]
