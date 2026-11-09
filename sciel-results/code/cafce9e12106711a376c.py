def construct():
    from itertools import combinations
    from math import lcm
    
    n = 128
    vertices = list(range(1, n+1))
    
    # Precompute bad triples
    bad_triples = []
    for a, b, c in combinations(vertices, 3):
        if lcm(a,b) == lcm(a,c) == lcm(b,c):
            bad_triples.append((a,b,c))
    
    # For each vertex, store set of pairs that form bad triples with it
    conflict_pairs = {v: set() for v in vertices}
    for (a,b,c) in bad_triples:
        conflict_pairs[a].add((b,c))
        conflict_pairs[b].add((a,c))
        conflict_pairs[c].add((a,b))
    
    # Degree
    degree = {v: len(conflict_pairs[v]) for v in vertices}
    ordered = sorted(vertices, key=lambda v: -degree[v])
    
    best = []
    
    def can_add(v, selected):
        for (a,b) in conflict_pairs[v]:
            if a in selected and b in selected:
                return False
        return True
    
    def maximal_matching(rem_set):
        used = set()
        count = 0
        for (a,b,c) in bad_triples:
            if a in rem_set and b in rem_set and c in rem_set:
                if a not in used and b not in used and c not in used:
                    count += 1
                    used.add(a)
                    used.add(b)
                    used.add(c)
        return count
    
    def solve(selected, remaining, idx):
        nonlocal best
        if not remaining:
            if len(selected) > len(best):
                best = list(selected)
            return
        
        rem_set = set(remaining)
        match_size = maximal_matching(rem_set)
        upper_bound = len(selected) + len(remaining) - match_size
        if upper_bound <= len(best):
            return
        
        if idx == len(ordered):
            if len(selected) > len(best):
                best = list(selected)
            return
        
        v = ordered[idx]
        rest = remaining
        
        # Option 1: include v if possible
        if v in rem_set and can_add(v, selected):
            selected.add(v)
            new_remaining = [w for w in rest if can_add(w, selected)]
            solve(selected, new_remaining, idx+1)
            selected.remove(v)
        
        # Option 2: exclude v
        solve(selected, rest, idx+1)
    
    all_remaining = list(ordered)
    solve(set(), all_remaining, 0)
    
    return best
