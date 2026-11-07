import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    N = 12
    LCM = 27720  # LCM(1,2,...,12)
    MAX_L = 144  # upper bound on f(12, m)
    
    def can_match(m, L):
        """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i using bitmask backtracking."""
        # For each i, candidate deltas d in [1, L] such that (m + d) % i == 0
        # i.e., d ≡ (-m) mod i
        cands = []
        for i in range(1, N + 1):
            r = (-m) % i
            # d in [1, L] with d % i == r
            # if r == 0, d = i, 2i, ...; else d = r, r+i, ...
            if r == 0:
                start = i
            else:
                start = r
            lst = []
            d = start
            while d <= L:
                lst.append(d)
                d += i
            cands.append(lst)
        
        # Sort by most constrained first
        cands.sort(key=lambda x: len(x))
        
        # If any has 0 candidates, fail immediately
        if len(cands[0]) == 0:
            return False
        
        # Bitmask backtracking
        # L <= 144, so we need 144 bits - use Python int
        def bt(k, used_mask):
            if k == N:
                return True
            lst = cands[k]
            for d in lst:
                bit = 1 << (d - 1)
                if not (used_mask & bit):
                    if bt(k + 1, used_mask | bit):
                        return True
            return False
        
        return bt(0, 0)
    
    def f_12(m):
        """Compute least positive L such that (m, m+L] works."""
        lo, hi = 1, MAX_L
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Exhaustive search over one period
    best_m = 0
    best_val = 0
    
    for r in range(LCM):
        if time.time() - start_time > deadline:
            break
        val = f_12(r)
        if val > best_val:
            best_val = val
            best_m = r
    
    return int(best_m)
