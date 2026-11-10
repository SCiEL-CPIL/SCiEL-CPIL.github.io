def construct():
    from math import lcm
    from itertools import combinations
    
    N = 128
    vertices = list(range(1, N + 1))
    
    # 1. Generate forbidden triples
    forbidden = []
    # Optimization: Only check triples where a,b,c are not too large?
    # No, just check all.
    for a, b, c in combinations(vertices, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            forbidden.append((a, b, c))
            
    if not forbidden:
        return vertices

    # 2. Solve Minimum Hitting Set
    # We want to find the smallest set R of vertices such that every triple in forbidden has at least one vertex in R.
    # The result will be vertices - R.
    
    best_removed = set()
    best_size = N # Upper bound: remove everything (though we can always keep 1)
    
    # Greedy initial bound
    remaining_triples_greedy = set(forbidden)
    greedy_removed = set()
    while remaining_triples_greedy:
        freq = {}
        for t in remaining_triples_greedy:
            for v in t:
                freq[v] = freq.get(v, 0) + 1
        if not freq:
            break
        max_v = max(freq, key=freq.get)
        greedy_removed.add(max_v)
        remaining_triples_greedy = {t for t in remaining_triples_greedy if max_v not in t}
        
    best_size = len(greedy_removed)
    best_removed = greedy_removed
    
    # DFS
    def dfs(remaining_triples, current_removed):
        nonlocal best_size, best_removed
        
        if not remaining_triples:
            if len(current_removed) < best_size:
                best_size = len(current_removed)
                best_removed = set(current_removed)
            return
            
        if len(current_removed) >= best_size:
            return
            
        # Heuristic: Pick triple with max sum of degrees
        deg = {}
        for t in remaining_triples:
            for v in t:
                deg[v] = deg.get(v, 0) + 1
                
        max_sum = -1
        target_t = None
        for t in remaining_triples:
            s = sum(deg.get(v, 0) for v in t)
            if s > max_sum:
                max_sum = s
                target_t = t
                
        if not target_t:
            return

        for v in target_t:
            if len(current_removed) + 1 >= best_size:
                continue
            new_current = current_removed | {v}
            new_remaining = {tr for tr in remaining_triples if v not in tr}
            dfs(new_remaining, new_current)

    dfs(set(forbidden), set())
    
    result = [v for v in vertices if v not in best_removed]
    return result
