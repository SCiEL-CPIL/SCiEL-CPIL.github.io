import time

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i using DFS."""
    # For each i, compute which positions j in 0..L-1 satisfy (m+1+j) % i == 0
    # i.e., j % i == (-(m+1)) % i
    options = []
    for i in range(1, n + 1):
        target = (-(m + 1)) % i
        mask = 0
        # Find all j in [0, L) where j % i == target
        # j = target, target+i, target+2i, ... < L
        start = target
        if start < 0:
            start += i
        # Actually target is in [0, i-1], so start = target
        for j in range(target, L, i):
            mask |= (1 << j)
        if mask == 0:
            return False
        options.append((mask.bit_count(), mask))
    
    # Sort by number of options (most constrained first)
    options.sort(key=lambda x: x[0])
    
    used = 0
    
    def dfs(k, used_mask):
        if k == n:
            return True
        count, mask = options[k]
        avail = mask & ~used_mask
        if avail == 0:
            return False
        # Try each available position
        while avail:
            bit = avail & (-avail)
            avail -= bit
            if dfs(k + 1, used_mask | bit):
                return True
        return False
    
    return dfs(0, 0)

def _f_val(m, n=12):
    """Find the least positive L such that (m, m+L] has a valid matching."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    best_m = 12
    best_f = _f_val(12)
    
    # Search strategy: try m values that are likely to maximize f
    # m near multiples of large numbers, or values that create "gaps"
    
    # First pass: try a set of promising candidates
    candidates = []
    
    # m such that m+1 is divisible by many numbers (makes matching harder for small L)
    # Actually we want m such that the interval (m, m+L] has "unlucky" distributions
    
    # Try m values around multiples of LCM subsets
    # For each subset of {1..12}, consider m where m+1 ≡ 0 mod lcm(subset)
    
    # Simpler: just scan a good set of m values
    # Focus on m where m+1 is divisible by 11 or 12 (most constrained)
    
    # Generate candidates: m = k*11 - 1, m = k*12 - 1, m = k*lcm - 1 for various k
    for base in range(1, 1000001):
        if base % 11 == 1:  # m+1 divisible by 11
            candidates.append(base - 1)
        if base % 12 == 1:  # m+1 divisible by 12
            candidates.append(base - 1)
    
    # Also try m where m+1 is divisible by 11*12=132
    for base in range(132, 1000001, 132):
        candidates.append(base - 1)
    
    # Add some random-ish spread
    for k in range(0, 1000001, 1000):
        candidates.append(k)
    
    # Remove duplicates, keep unique
    candidates = list(set(candidates))
    candidates.sort()
    
    # Evaluate candidates, prioritizing those likely to give high f
    # Sort by heuristic: m+1 divisible by more of {1..12}
    def heuristic(m):
        mp1 = m + 1
        score = 0
        for i in range(1, 13):
            if mp1 % i == 0:
                score += i
        return -score  # More divisible = more negative = higher priority
    
    candidates.sort(key=heuristic)
    
    evaluated = 0
    for m in candidates:
        if time.time() > deadline:
            break
        if m < 0 or m > 1000000:
            continue
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
            if best_f == 144:
                break
        evaluated += 1
    
    return int(best_m)
