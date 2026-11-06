"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 24
    elements = list(range(1, n + 1))
    
    # Precompute conflict constraints
    conflicts = {}
    for a in elements:
        conflicts[a] = []
        others = [x for x in elements if x != a]
        target = Fraction(1, a)
        
        for b, c in combinations(others, 2):
            if Fraction(1, b) + Fraction(1, c) == target:
                conflicts[a].append(frozenset([b, c]))
        
        for b, c, d in combinations(others, 3):
            if Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                conflicts[a].append(frozenset([b, c, d]))
    
    def check_add(S, x):
        new_S = S | {x}
        for conflict in conflicts[x]:
            if conflict.issubset(new_S):
                return False
        for a in S:
            for conflict in conflicts[a]:
                if x in conflict and conflict.issubset(new_S):
                    return False
        return True
    
    best = set()
    best_size = 0
    
    def dfs(S, remaining):
        nonlocal best, best_size
        if time.time() - start_time > DEADLINE:
            return
        
        if len(S) + len(remaining) <= best_size:
            return
        
        if not remaining:
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
            return
        
        x = remaining[0]
        rest = remaining[1:]
        
        if check_add(S, x):
            dfs(S | {x}, rest)
        
        dfs(S, rest)
    
    dfs(set(), elements)
    
    if not best:
        return list(range(13, 25))
    return list(best)

# EVOLVE-BLOCK-END
