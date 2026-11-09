"""Find N ≤ 10^18 with many representations as sum of two powerful numbers."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict
import bisect

def _is_powerful(n):
    """Check if n is powerful (every prime factor appears with exponent >= 2)."""
    if n == 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp == 1:
                return False
        p += 1 if p == 2 else 2
    if temp > 1:
        return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit.
    A powerful number has form a^2 * b^3 where b is squarefree."""
    result = set()
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        if not _is_squarefree(b):
            continue
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                result.add(val)
    return sorted(result)

def _is_squarefree(n):
    """Check if n is squarefree."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % (p * p) == 0:
            return False
        p += 1 if p == 2 else 2
    return True

def _count_representations(N, powerful_list, powerful_set):
    """Count how many pairs (a, b) with a <= b, a+b=N, both powerful."""
    count = 0
    half = N // 2
    # Only need to check a <= N/2
    for a in powerful_list:
        if a > half:
            break
        b = N - a
        if b in powerful_set:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin
    
    # Generate powerful numbers up to a moderate limit for fast lookup
    # 10^7 gives ~3000 numbers, 10^8 gives ~10000, 10^9 gives ~30000
    pw_limit = 10**8
    powerful = _generate_powerful(pw_limit)
    powerful_set = set(powerful)
    n_pw = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy: Search for N with many representations
    # Phase 1: Use sumset counting for small N (up to 2*pw_limit) to find dense regions
    # Phase 2: Probe larger N values using the powerful set
    
    best_N = 2
    best_count = 1  # N=2: (1,1)
    
    # Phase 1: Sumset counting for N up to 2*pw_limit
    # This is O(n_pw^2) but n_pw ~ 10000, so ~50M pairs - might be tight
    # Let's use a smaller limit for the sumset phase
    pw_limit_small = 10**7
    powerful_small = _generate_powerful(pw_limit_small)
    powerful_small_set = set(powerful_small)
    n_small = len(powerful_small)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 40:
        # Not enough time for sumset, go straight to targeted search
        pass
    else:
        # Sumset counting for small range
        sum_count = defaultdict(int)
        max_N_small = 2 * pw_limit_small
        
        for i, a in enumerate(powerful_small):
            if a > max_N_small - a:
                break
            b_min = a
            b_max = max_N_small - a
            lo = bisect.bisect_left(powerful_small, b_min)
            hi = bisect.bisect_right(powerful_small, b_max)
            for j in range(lo, hi):
                b = powerful_small[j]
                s = a + b
                sum_count[s] += 1
        
        if sum_count:
            # Find top candidates from sumset
            # Sort by count descending, take top few
            sorted_sums = sorted(sum_count.items(), key=lambda x: -x[1])
            top_candidates = [s for s, c in sorted_sums[:50] if c > best_count]
            
            # Verify top candidates with the larger powerful set
            for N in top_candidates:
                if N > 2 * pw_limit:
                    continue
                cnt = _count_representations(N, powerful, powerful_set)
                if cnt > best_count:
                    best_count = cnt
                    best_N = N
        
        elapsed = time.time() - start_time
    
    # Phase 2: Targeted search for larger N values
    # Powerful numbers have density ~ c*sqrt(x) for large x
    # So for large N, the number of representations should grow like sqrt(N)
    # Let's probe N values at various scales
    
    # Generate candidate N values to probe
    # Focus on N that are sums of small powerful numbers (likely dense)
    # Also probe geometrically spaced N values
    
    candidate_Ns = set()
    
    # Add N values from sum of pairs of small powerful numbers
    # Sample a subset of pairs to get candidate N values
    if n_small > 100:
        # Take first K powerful numbers and compute all pair sums
        K = min(200, n_small)
        for i in range(K):
            for j in range(i, K):
                s = powerful_small[i] + powerful_small[j]
                if s <= 2 * pw_limit:
                    candidate_Ns.add(s)
    
    # Add geometrically spaced N values
    for exp in range(1, 18):
        base = 10 ** exp
        for mult in [1, 2, 3, 5, 7, 10, 15, 20]:
            N = base * mult
            if N <= 2 * pw_limit and N >= 2:
                candidate_Ns.add(N)
    
    # Add multiples of best_N found so far
    if best_N > 2:
        for mult in range(2, 20):
            N = best_N * mult
            if N <= 2 * pw_limit:
                candidate_Ns.add(N)
    
    # Evaluate candidates
    # Sort by likelihood: prefer N in dense regions
    # Process candidates, but limit total work
    max_evals = 500
    evals_done = 0
    
    # Prioritize candidates: those near best_N, then smaller N (denser)
    candidate_list = sorted(candidate_Ns, key=lambda x: (abs(x - best_N), x))
    
    for N in candidate_list:
        if N < 2 or N > 2 * pw_limit:
            continue
        if N == best_N:
            continue
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        if evals_done >= max_evals:
            break
        
        # Quick estimate: count representations
        cnt = _count_representations(N, powerful, powerful_set)
        evals_done += 1
        
        if cnt > best_count:
            best_count = cnt
            best_N = N
    
    # Phase 3: If we have time, try to improve by searching around best_N
    elapsed = time.time() - start_time
    if elapsed < deadline - 20 and best_N > 2:
        # Search in a window around best_N
        window = max(1000, best_N // 100)
        for offset in range(-window, window + 1):
            N = best_N + offset
            if N < 2 or N > 2 * pw_limit:
                continue
            if N == best_N:
                continue
            
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            
            cnt = _count_representations(N, powerful, powerful_set)
            if cnt > best_count:
                best_count = cnt
                best_N = N
    
    # Build pairs for best_N
    pairs = []
    half = best_N // 2
    for a in powerful:
        if a > half:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
    
    # Verify
    assert best_N >= 2
    assert len(pairs) == best_count
    for a, b in pairs:
        assert a <= b
        assert a + b == best_N
        assert _is_powerful(a)
        assert _is_powerful(b)
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
