import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    n = 12
    period = 27720  # lcm(1..12)
    max_L = 144

    # Precompute: for each i in 1..12 and each residue r in 0..i-1,
    # the bitmask of d in 1..max_L where d % i == r
    full_mask = {}
    for i in range(1, n + 1):
        full_mask[i] = [0] * i
        for d in range(1, max_L + 1):
            rem = d % i
            full_mask[i][rem] |= (1 << d)

    def has_perfect_matching(m, L):
        """Check if (m, m+L] has a perfect matching for distinct multiples."""
        if L < n:
            return False

        adj_masks = [0] * n
        counts = [0] * n
        union_mask = 0

        for i in range(1, n + 1):
            rem = (-m) % i
            mask = full_mask[i][rem] & ((1 << (L + 1)) - 1)
            adj_masks[i - 1] = mask
            cnt = mask.bit_count()
            counts[i - 1] = cnt
            union_mask |= mask

        for c in counts:
            if c == 0:
                return False

        if union_mask.bit_count() < n:
            return False

        # Hall's condition for pairs
        for a in range(n):
            for b in range(a + 1, n):
                if (adj_masks[a] | adj_masks[b]).bit_count() < 2:
                    return False

        # Order by count ascending (most constrained first)
        order = sorted(range(n), key=lambda idx: counts[idx])

        matchR = [-1] * (L + 1)

        def dfs(idx, seen_mask):
            mask = adj_masks[idx] & ~seen_mask
            while mask:
                d = (mask & (-mask)).bit_length() - 1
                mask &= mask - 1
                if matchR[d] == -1:
                    matchR[d] = idx
                    return True
                if dfs(matchR[d], seen_mask | (1 << d)):
                    matchR[d] = idx
                    return True
            return False

        for orig_idx in order:
            if not dfs(orig_idx, 0):
                return False
        return True

    def compute_L(m):
        """Binary search for minimum L for given m."""
        lo, hi = 1, max_L
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_perfect_matching(m, mid):
                hi = mid - 1
            else:
                lo = mid + 1
        return lo  # lo is the minimum L

    best_m = 0
    best_L = 0

    # Generate high-potential candidates using CRT-inspired reasoning
    # For large L, we want m such that for many i, the next multiple is far away
    # This happens when m is just below multiples of several i's
    
    candidates = set()
    
    # Strategy 1: m values where m ≡ -1 mod i for small i (1-6)
    # This forces b_i = m+i to be the first multiple, spreading out the matches
    for target_residues in range(2, 7):
        # Find m in [0, period) such that m ≡ -1 mod i for i=1..target_residues
        # Using CRT step by step
        m_val = 0
        modulus = 1
        for i in range(2, target_residues + 1):
            # Solve m ≡ -1 mod i, current m ≡ m_val mod modulus
            # m = m_val + k * modulus, need m_val + k*modulus ≡ -1 mod i
            # k*modulus ≡ -1 - m_val mod i
            a = modulus % i
            b = (-1 - m_val) % i
            if gcd(a, i) != gcd(a, b):
                break
            g = gcd(a, i)
            a_div = a // g
            b_div = b // g
            i_div = i // g
            # Find inverse of a_div mod i_div
            try:
                inv = pow(a_div, -1, i_div)
            except:
                break
            k = (b_div * inv) % i_div
            m_val = m_val + k * modulus
            modulus *= i_div
            m_val %= modulus
        
        if m_val < period:
            # Add all solutions in [0, period)
            for k in range(period // modulus + 1):
                m_candidate = m_val + k * modulus
                if m_candidate < period:
                    candidates.add(m_candidate)

    # Strategy 2: m values that are just below lcm of subsets
    # m = lcm(1..k) - 1 for various k
    from math import prod
    lcm_val = 1
    for k in range(2, 13):
        # Compute lcm(1..k)
        lcm_val = lcm_val * k // gcd(lcm_val, k)
        if lcm_val > period:
            break
        m_candidate = (lcm_val - 1) % period
        candidates.add(m_candidate)
        # Also try m = lcm_val - 1 - j for small j
        for j in range(1, 20):
            candidates.add((lcm_val - 1 - j) % period)

    # Strategy 3: m values that are multiples of small numbers minus 1
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        for k in range(1, period // base + 1):
            m_candidate = (k * base - 1) % period
            candidates.add(m_candidate)

    # Strategy 4: Uniformly spaced candidates to ensure coverage
    step = max(1, period // 500)
    for m in range(0, period, step):
        candidates.add(m)

    # Strategy 5: Specifically target m values where many i have their first multiple at different distances
    # For i=2,3,4,5,6,7,8,9,10,11,12, the first multiple > m is at distance ((m//i)+1)*i - m
    # We want these distances to be as spread out as possible and large
    
    # Add candidates near where multiple "gaps" align
    for offset in range(0, 144):
        candidates.add(offset)
        candidates.add(period - 1 - offset)

    # Sort candidates by a heuristic score: how "spread out" the first multiples are
    def heuristic_score(m):
        """Higher score = more promising for large L."""
        gaps = []
        for i in range(1, 13):
            next_mult = ((m // i) + 1) * i
            gap = next_mult - m
            gaps.append(gap)
        # We want large gaps and good spread
        # Use sum of gaps and variance
        avg_gap = sum(gaps) / len(gaps)
        variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
        # Penalize small gaps heavily
        min_gap = min(gaps)
        return (min_gap, variance, sum(gaps))
    
    candidates = sorted(candidates, key=heuristic_score, reverse=True)

    # Evaluate top candidates first
    evaluated = set()
    
    for m in candidates:
        if time.time() - start_time > deadline - 5.0:
            break
        
        if m in evaluated:
            continue
        evaluated.add(m)
        
        L_val = compute_L(m)
        
        if L_val > best_L:
            best_L = L_val
            best_m = m
            
            # If we've found a very high L, we can be more aggressive with time
            if best_L >= 100:
                # Continue evaluating but with tighter time check
                pass

    # If we haven't found a great candidate, do a full scan as fallback
    if best_L < 80 and time.time() - start_time < deadline - 10.0:
        for m in range(0, period):
            if time.time() - start_time > deadline - 5.0:
                break
            
            if m in evaluated:
                continue
            evaluated.add(m)
            
            L_val = compute_L(m)
            
            if L_val > best_L:
                best_L = L_val
                best_m = m
                
                if best_L >= max_L:
                    break

    # Local refinement around best_m
    if time.time() - start_time < deadline - 2.0 and best_L < max_L:
        window = 50
        for dm in range(-window, window + 1):
            if dm == 0:
                continue
            m2 = (best_m + dm) % period
            
            if m2 in evaluated:
                continue
            evaluated.add(m2)
            
            L_val = compute_L(m2)
            
            if L_val > best_L:
                best_L = L_val
                best_m = m2

    # Verify and return
    if best_m < 0 or best_m > 1000000:
        best_m = 0

    return int(best_m)
