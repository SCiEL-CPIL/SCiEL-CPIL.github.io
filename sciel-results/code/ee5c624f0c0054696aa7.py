"""Rewritten program for equal-lcm-triple avoidance."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    deadline = time.time() + 85  # safety margin under 90s
    
    n = 128
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c and lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, n + 1))
    
    # Greedy vertex cover on 3-uniform hypergraph
    # Start with all elements selected
    selected = set(range(1, n + 1))
    # For each element, count how many bad triples it's in
    # We'll iteratively remove the element that covers the most remaining bad triples
    
    # Build adjacency: for each element, list of triple indices it belongs to
    elem_to_triples = {i: [] for i in range(1, n + 1)}
    for idx, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)
    
    # Active triples
    active = set(range(len(bad_triples)))
    
    # Count how many active triples each element covers
    cover_count = {}
    for e in range(1, n + 1):
        cover_count[e] = sum(1 for t in elem_to_triples[e] if t in active)
    
    removed = set()
    
    while active:
        if time.time() > deadline:
            break
        
        # Find element with max cover_count among non-removed
        best_elem = None
        best_count = -1
        for e in range(1, n + 1):
            if e not in removed:
                if cover_count[e] > best_count:
                    best_count = cover_count[e]
                    best_elem = e
        
        if best_elem is None or best_count <= 0:
            break
        
        # Remove best_elem
        removed.add(best_elem)
        
        # Update: remove all triples containing best_elem from active
        triples_to_remove = [t for t in elem_to_triples[best_elem] if t in active]
        for t in triples_to_remove:
            active.discard(t)
            # Update cover counts for other elements in this triple
            a, b, c = bad_triples[t]
            for other in (a, b, c):
                if other != best_elem and other not in removed:
                    cover_count[other] -= 1
    
    result = [i for i in range(1, n + 1) if i not in removed]
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
