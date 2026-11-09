"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations
from math import lcm

def construct():
    n = 128
    universe = list(range(1, n + 1))
    
    # Precompute LCMs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[(i, j)] = lcm(i, j)
    
    # Find all bad triples
    bad_triples = []
    for a, b, c in combinations(universe, 3):
        if lcm_table[(a,b)] == lcm_table[(a,c)] == lcm_table[(b,c)]:
            bad_triples.append((a, b, c))
    
    # For each vertex, list of bad triples it participates in
    vertex_triples = {v: [] for v in universe}
    for idx, (a,b,c) in enumerate(bad_triples):
        vertex_triples[a].append(idx)
        vertex_triples[b].append(idx)
        vertex_triples[c].append(idx)
    
    # Greedy solution for initial lower bound
    selected = set()
    for x in universe:
        ok = True
        for tidx in vertex_triples[x]:
            a,b,c = bad_triples[tidx]
            others = [v for v in (a,b,c) if v != x]
            if all(o in selected for o in others):
                ok = False
                break
        if ok:
            selected.add(x)
    
    best_size = len(selected)
    best_set = set(selected)
    
    sys.setrecursionlimit(20000)
    
    def can_add(v, chosen):
        for tidx in vertex_triples[v]:
            a,b,c = bad_triples[tidx]
            others = [x for x in (a,b,c) if x != v]
            if all(o in chosen for o in others):
                return False
        return True
    
    # Branch and bound
    # Order vertices by "constraint" - those in more bad triples first for better pruning
    vertex_order = sorted(universe, key=lambda v: len(vertex_triples[v]), reverse=True)
    
    def solve(chosen, remaining_set, idx):
        nonlocal best_size, best_set
        
        if idx == len(vertex_order):
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)
            return
        
        # Upper bound: current + remaining count
        if len(chosen) + (len(vertex_order) - idx) <= best_size:
            return
        
        v = vertex_order[idx]
        
        # Try include v
        if can_add(v, chosen):
            chosen.add(v)
            solve(chosen, remaining_set, idx + 1)
            chosen.discard(v)
        
        # Try exclude v
        solve(chosen, remaining_set, idx + 1)
    
    solve(set(), set(universe), 0)
    
    return list(best_set) if best_set else [1]

# EVOLVE-BLOCK-END
