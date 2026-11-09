import time
import itertools
import random
from collections import Counter

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
    chosen = []
    remaining = set(all_edges)
    
    round_i = 0
    while remaining and _deadline_ok() and round_i < max_rounds:
        round_i += 1
        rem_mask = _adj_mask(n, remaining)
        maxcls = _maximal_cliques_bit(rem_mask, n, 3, 500000, _deadline_ok)
        
        if not maxcls:
            # No clique of size >= 3, pick largest clique possible (size 2)
            # Find any edge
            if remaining:
                e = next(iter(remaining))
                chosen.append([e[0], e[1]])
                remaining.remove(e)
            continue
        
        # Score each clique: prefer larger cliques, then those covering "rare" edges
        # For efficiency, just sort by size descending, then by lexicographic order
        maxcls.sort(key=lambda m: (-m.bit_count(), m))
        
        # Select the largest maximal clique
        best_clique_mask = maxcls[0]
        verts = list(_iter_bits(best_clique_mask))
        
        if len(verts) >= 2:
            ec = _clique_edges_mask(best_clique_mask)
            if ec <= remaining:
                remaining -= ec
                chosen.append(verts)
            else:
                # Should not happen, but fallback
                break
        else:
            break

    # Add remaining edges as individual cliques
    for e in remaining:
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
