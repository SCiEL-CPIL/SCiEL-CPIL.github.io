import time

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses bipartite matching (augmenting path / Hopcroft-Karp simplified for small n).
    """
    # For each i in 1..n, find which positions in (m, m+L] are divisible by i
    # Position j (0-indexed) corresponds to value m+1+j
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        # Find all j in [0, L) such that (m+1+j) % i == 0
        # m+1+j ≡ 0 (mod i) => j ≡ -(m+1) (mod i)
        # j ranges from 0 to L-1
        rem = (- (m + 1)) % i
        # j = rem, rem+i, rem+2i, ... < L
        if rem >= L:
            return False
        for j in range(rem, L, i):
            adj[i-1].append(j)
        if not adj[i-1]:
            return False
    
    # Use augmenting path matching (Kuhn's algorithm)
    # match[j] = i-1 means position j is matched to row i-1
    match = [-1] * L
    
    def try_augment(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match[v] == -1 or try_augment(match[v], visited):
                match[v] = u
                return True
        return False
    
    count = 0
    for u in range(n):
        visited = [False] * L
        if try_augment(u, visited):
            count += 1
    
    return count == n

def _f_val(m, n=12):
    """
    Find the least positive L such that (m, m+L] contains a valid matching.
    Uses binary search since the property is monotone in L.
    """
    # Max L is stated to be at most 144
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    """
    Find m in 0..1000000 that maximizes f(12, m).
    """
    # The function f(12, m) is periodic with period lcm(1..12) = 27720
    # So we only need to search m in [0, 27720)
    K = 27720
    best_m = 0
    best_L = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds margin (90s limit)
    
    for m in range(K):
        if time.time() > deadline:
            break
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
            # If we hit the theoretical max, we can stop early
            if best_L == 144:
                break
                
    return int(best_m)
