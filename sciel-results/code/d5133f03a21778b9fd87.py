"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd

def _prime_factors(n):
    if n == 1:
        return frozenset()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return frozenset(factors)

def _build_data(n_max=256):
    pf = [frozenset() for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        pf[i] = _prime_factors(i)
    
    conflict = [set() for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i + 1, n_max + 1):
            if pf[i] & pf[j]:
                conflict[i].add(j)
                conflict[j].add(i)
    
    weights = [0] + list(range(1, n_max + 1))
    return conflict, weights

def _color_bound(candidates, conflict, weights):
    if not candidates:
        return 0
    verts = sorted(candidates, key=lambda v: weights[v], reverse=True)
    color_of = {}
    color_max = {}
    for v in verts:
        used = set()
        for u in conflict[v]:
            if u in color_of:
                used.add(color_of[u])
        c = 0
        while c in used:
            c += 1
        color_of[v] = c
        if c not in color_max or weights[v] > color_max[c]:
            color_max[c] = weights[v]
    return sum(color_max.values())

def _solve(conflict, weights, n_max):
    best = [0]
    best_set = [set()]
    
    def dfs(candidates, selected, cur_sum):
        if cur_sum > best[0]:
            best[0] = cur_sum
            best_set[0] = set(selected)
        
        if not candidates:
            return
        
        # Upper bound
        ub = cur_sum + _color_bound(candidates, conflict, weights)
        if ub <= best[0]:
            return
        
        # Pick vertex with max weight
        v = max(candidates, key=lambda x: weights[x])
        
        # Branch 1: include v
        selected.add(v)
        neighbors = conflict[v]
        new_cands = frozenset(u for u in candidates if u != v and u not in neighbors)
        dfs(new_cands, selected, cur_sum + weights[v])
        selected.remove(v)
        
        # Branch 2: exclude v
        new_cands2 = frozenset(candidates - {v})
        dfs(new_cands2, selected, cur_sum)
    
    initial = frozenset(range(2, n_max + 1))  # 1 is always included
    dfs(initial, {1}, 1)
    return best_set[0]

def construct():
    n_max = 256
    conflict, weights = _build_data(n_max)
    result_set = _solve(conflict, weights, n_max)
    result = sorted(result_set)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
