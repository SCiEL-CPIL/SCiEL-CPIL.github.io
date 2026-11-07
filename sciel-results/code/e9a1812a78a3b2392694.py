import time

_LCM = 27720

def _can_match_fast(m, L):
    """Fast check using Hall's condition with precomputation."""
    # For each i in 1..12, multiples in (m, m+L]:
    # count_i = number of multiples of i in (m, m+L]
    # Hall: for all S, |N(S)| >= |S|
    # N(S) = {v in (m,m+L] : some i in S divides v}
    
    # Since L <= 144, enumerate all values in window
    # For each value, bitmask of which i divide it
    # Then for each subset S, count values with (mask & S) != 0
    
    # Precompute for each value in (m, m+L] its divisor mask
    # At most 144 values, 12 bits each
    
    masks = []
    for v in range(m + 1, m + L + 1):
        mask = 0
        for i in range(1, 13):
            if v % i == 0:
                mask |= (1 << (i - 1))
        masks.append(mask)
    
    # For each of 4096 subsets, count values where mask & subset != 0
    # But we can do this incrementally or just brute force
    # 4096 * 144 = 590k - acceptable
    
    n = len(masks)
    # For subset S (bitmask), |N(S)| = count of masks where (mask & S) != 0
    # Equivalently, n - count of masks where (mask & S) == 0
    
    # Precompute: for each value's mask, which subsets does it NOT cover?
    # mask & S == 0 means S is subset of (~mask) & 0xFFF
    # So for each value with mask M, it fails to be in N(S) for all S subset of (~M) & 0xFFF
    
    # fail_count[S] = number of values not in N(S)
    # We need fail_count[S] <= n - |S| for all S
    
    # Initialize fail_count
    fail_count = [0] * 4096
    
    for M in masks:
        # This value is NOT in N(S) for all S where S & M == 0
        # i.e., S is a subset of complement of M within 12 bits
        comp = (~M) & 0xFFF
        # Iterate all subsets of comp
        sub = comp
        while True:
            fail_count[sub] += 1
            if sub == 0:
                break
            sub = (sub - 1) & comp
    
    # Check Hall: for all S with |S| > 0, fail_count[S] <= n - |S|
    # i.e., |N(S)| = n - fail_count[S] >= |S|
    
    for S in range(1, 4096):
        # popcount of S
        pc = bin(S).count('1')
        if fail_count[S] > n - pc:
            return False
    
    return True

def _f(m):
    """Compute f(12, m)."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_fast(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 80
    
    best_m = 0
    best_f = 0
    
    # Check all 27720 residues
    # Each _f does ~8 binary search steps, each with Hall check
    # Hall check: 4096 subsets * popcount + building fail_count
    # Building fail_count: for each of L values, iterate subsets of complement
    # Worst case complement has all 12 bits, so 4096 subsets per value
    # L up to 144: 144 * 4096 = 590k per _can_match
    # 8 * 590k = 4.7M per _f
    # 27720 * 4.7M = 130B... way too slow
    
    # Need much faster approach. Let me reconsider.
    
    # Actually, for small L, the number of values is small.
    # For L=1: 1 value, fail_count building is trivial
    # The binary search will have varying L values
    
    # Let me just try it and see if it's fast enough with early termination
    
    # Alternative: use a much simpler sufficient condition
    # If for each i, count of multiples of i in (m, m+L] >= 1, AND
    # the system is "loose enough", matching likely exists.
    # But we need exact answer.
    
    # Let me try a different approach: since L is small (<=144),
    # and we have only 12 left nodes, use a simple DFS matching
    # which should be very fast in practice.
    
    def _can_match_dfs(m, L):
        adj = [[] for _ in range(12)]
        for i in range(1, 13):
            first = ((m // i) + 1) * i
            v = first
            while v <= m + L:
                adj[i-1].append(v)
                v += i
        
        for i in range(12):
            if not adj[i]:
                return False
        
        match_val = {}
        
        def dfs(i, visited):
            for v in adj[i]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_val or dfs(match_val[v], visited):
                    match_val[v] = i
                    return True
            return False
        
        for i in range(12):
            visited = set()
            if not dfs(i, visited):
                return False
        return True
    
    def _f2(m):
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if _can_match_dfs(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Test speed
    import time as _t
    t0 = _t.time()
    for r in range(100):
        _f2(r)
    elapsed = _t.time() - t0
    per_call = elapsed / 100
    total_est = per_call * 27720
    
    if total_est < 70:
        # Can check all residues
        for r in range(27720):
            if _t.time() > deadline:
                break
            f_val = _f2(r)
            if f_val > best_f:
                best_f = f_val
                best_m = r
    else:
        # Coarse search
        step = max(1, int(27720 / (70 / per_call)))
        for r in range(0, 27720, step):
            if _t.time() > deadline:
                break
            f_val = _f2(r)
            if f_val > best_f:
                best_f = f_val
                best_m = r
        
        # Refine around best
        if best_m > 0:
            for r in range(max(0, best_m - step), min(27720, best_m + step + 1)):
                if _t.time() > deadline:
                    break
                f_val = _f2(r)
                if f_val > best_f:
                    best_f = f_val
                    best_m = r
    
    return int(best_m)
