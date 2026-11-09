def construct():
    import time
    from math import lcm
    
    deadline = time.time() + 80
    n = 128
    
    # Precompute LCM table
    lcm_t = [[0]*(n+1) for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(1, n+1):
            lcm_t[a][b] = lcm(a, b)
    
    # bad_pairs[x] = frozenset of pairs (a,b) with a<b such that (a,b,x) is bad
    bad_pairs = [frozenset() for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            L = lcm_t[a][b]
            for c in range(1, n+1):
                if c == a or c == b:
                    continue
                if lcm_t[a][c] == L and lcm_t[b][c] == L:
                    bad_pairs[c] = bad_pairs[c] | frozenset([(a,b)])
                    bad_pairs[a] = bad_pairs[a] | frozenset([(b,c)])
                    bad_pairs[b] = bad_pairs[b] | frozenset([(a,c)])
    
    selected = set()
    remaining = set(range(1, n+1))
    
    while remaining and time.time() < deadline - 10:
        best_x = None
        best_conf = float('inf')
        for x in sorted(remaining):
            conf = 0
            bp = bad_pairs[x]
            for a in selected:
                for b in selected:
                    if a < b and (a,b) in bp:
                        conf += 1
            if conf < best_conf:
                best_conf = conf
                best_x = x
            elif conf == best_conf and (best_x is None or x < best_x):
                best_x = x
            if conf == 0:
                break
        if best_x is None:
            break
        selected.add(best_x)
        remaining.discard(best_x)
    
    return sorted(selected)
