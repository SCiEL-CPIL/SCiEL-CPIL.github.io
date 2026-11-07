def min_span(m):
    # Find least L such that (m, m+L] has distinct b_i div by i
    # Max L is bounded, say 150 (task says at most 144)
    
    def exists_match(L):
        match = {} # k -> i
        
        def try_match(i, visited):
            for k in range(1, L + 1):
                if (m + k) % i != 0: continue
                if visited[k]: continue
                visited[k] = True
                if k not in match or try_match(match[k], visited):
                    match[k] = i
                    return True
            return False
            
        for i in range(1, 13):
            visited = [False] * (L + 1)
            if not try_match(i, visited):
                return False
        return True

    lo, hi = 1, 150
    # Ensure hi is a valid bound
    if not exists_match(hi):
        return hi
        
    while lo < hi:
        mid = (lo + hi) // 2
        if exists_match(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    best_m = 0
    best_val = 0
    # Search the full period of the divisibility pattern
    # lcm(1..12) = 27720
    for m in range(27720):
        val = min_span(m)
        if val > best_val:
            best_val = val
            best_m = m
    return best_m
