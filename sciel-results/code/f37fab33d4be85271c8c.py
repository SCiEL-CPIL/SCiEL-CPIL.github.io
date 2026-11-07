import time

def construct():
    start_time = time.time()
    deadline = 80.0

    N = 12
    PERIOD = 27720  # lcm(1,2,...,12)
    MAX_L = 144

    # Precompute candidate bitmasks for each (i, r) where r = m % i
    # mask[i][r] has bit (d-1) set if d is a valid offset for i given m ≡ r (mod i)
    # We only need d up to MAX_L
    cand_mask_table = [[0] * i for i in range(N + 1)]
    for i in range(1, N + 1):
        for r in range(i):
            d = (-r) % i
            if d == 0:
                d = i
            mask = 0
            while d <= MAX_L:
                mask |= (1 << (d - 1))
                d += i
            cand_mask_table[i][r] = mask

    # Precompute popcount for all possible masks up to MAX_L bits
    # For a mask of up to 144 bits, we use int.bit_count() which is fast
    
    def has_matching(m, L):
        """Check if there exists a matching of size 12 where i divides b_i
        and all b_i are in (m, m+L]."""
        # Build candidate masks limited to L
        cands = [0] * (N + 1)
        limit_mask = (1 << L) - 1
        for i in range(1, N + 1):
            r = m % i
            full_mask = cand_mask_table[i][r]
            cands[i] = full_mask & limit_mask
            if cands[i] == 0:
                return False
        
        # Sort by popcount for most-constrained-first
        order = list(range(1, N + 1))
        order.sort(key=lambda i: cands[i].bit_count())
        
        # Backtracking with bitsets and forward checking
        used = 0
        
        def bt(idx):
            nonlocal used
            if idx == N:
                return True
            i = order[idx]
            avail = cands[i] & ~used
            while avail:
                lowbit = avail & (-avail)
                avail ^= lowbit
                used |= lowbit
                # Forward check: make sure all remaining variables have at least one candidate
                feasible = True
                for j in range(idx + 1, N):
                    if (cands[order[j]] & ~used) == 0:
                        feasible = False
                        break
                if feasible and bt(idx + 1):
                    return True
                used ^= lowbit
            return False
        
        return bt(0)

    # Optimize: instead of binary search, for each residue, try increasing L
    # from a lower bound up to MAX_L, stopping at the first L that works.
    # But binary search is better for finding the minimum L.
    
    # Even better: precompute for each residue the minimum L by trying
    # candidate L values. Since MAX_L is only 144, and we have 27720 residues,
    # we need efficiency.
    
    # Strategy: For each residue r, binary search for the minimum L.
    # We can speed this up by noting that if L works for m, then any L' > L also works.
    # So binary search is optimal.
    
    # Additional speedup: cache results of has_matching calls
    match_cache = {}
    
    def cached_has_matching(m, L):
        key = (m, L)
        if key in match_cache:
            return match_cache[key]
        result = has_matching(m, L)
        match_cache[key] = result
        return result

    best_m = 0
    best_L = 0
    
    for r in range(PERIOD):
        if time.time() - start_time > deadline:
            break
        
        # Binary search for minimum L
        lo, hi = 1, MAX_L
        while lo < hi:
            mid = (lo + hi) // 2
            if cached_has_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1
        
        if lo > best_L:
            best_L = lo
            best_m = r
            if best_L == MAX_L:
                break
    
    return int(best_m)
