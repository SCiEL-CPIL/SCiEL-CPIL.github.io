"""Find N ≤ 10^18 with many representations as sum of two powerful numbers."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

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
    # temp is now 1 or a prime
    if temp > 1:
        return False  # remaining prime factor has exponent 1
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit.
    A powerful number has form a^2 * b^3 where b is squarefree."""
    result = set()
    # b must be squarefree and b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        # Check b is squarefree
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Phase 1: Generate powerful numbers up to a reasonable limit
    # Powerful numbers up to 10^9: roughly sqrt(10^9) * constant ≈ 30000-50000
    # Let's try 10^9 first
    pw_limit = 10**9
    powerful = _generate_powerful(pw_limit)
    powerful_set = set(powerful)
    n_pw = len(powerful)
    
    # Phase 2: For each powerful number a, we want to find N such that N - a is also powerful
    # Strategy: For a candidate N, count pairs (a, b) where a + b = N, a, b powerful, a <= b
    # 
    # Better approach: Build a frequency map of sums
    # For each pair (a, b) with a <= b, a + b = N, count it
    # But O(n_pw^2) is too large for 30000^2 = 900M
    #
    # Alternative: For each a in powerful, for each b in powerful with a <= b,
    # compute s = a + b and count[s] += 1
    # But we need to be smart about which pairs to check
    #
    # Key insight: We want N up to 10^18, but a, b <= 10^9 means a + b <= 2*10^9
    # So we can only find N up to 2*10^9 this way
    #
    # To get larger N, we need larger powerful numbers. But generating powerful numbers
    # up to 10^18 is infeasible (there are about sqrt(10^18) = 10^9 of them)
    #
    # Alternative strategy: Use the structure of powerful numbers
    # A powerful number is a^2 * b^3. For large N, we can have one large powerful number
    # and one small one.
    #
    # Let me think differently. The maximum N we can handle with the sumset approach
    # is 2 * pw_limit. To maximize representations, we want N in a dense region.
    #
    # For N <= 2 * 10^9, let's do the full sumset count
    # n_pw ≈ 30000, so n_pw^2 / 2 ≈ 450M operations - might be too slow
    #
    # Let's use a smarter approach: for each a, the valid b range is [a, N-a]
    # and we need b in powerful_set. We can binary search or use the sorted list.
    #
    # Actually, let's just do it with the smaller set and see how many we get.
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        # Not enough time, return best effort
        # Just return a simple result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Count representations for N values
    # For efficiency, only consider N up to 2 * pw_limit
    max_N = 2 * pw_limit
    
    # Use a dictionary to count sums
    # But 2*10^9 entries is too many. Instead, we'll sample candidate N values.
    #
    # Better: iterate over pairs but limit the work
    # For each a in powerful (small ones), for each b in powerful with a <= b:
    #   s = a + b
    #   if s <= max_N: count[s] += 1
    #
    # The number of pairs with a + b <= 2 * 10^9 where a, b <= 10^9 is about n_pw^2/2
    # which is ~450M for n_pw=30000. Too slow.
    #
    # Let's reduce pw_limit to make this feasible.
    # If pw_limit = 10^7, n_pw ≈ 3000, pairs ≈ 4.5M - feasible!
    # If pw_limit = 2*10^7, n_pw ≈ 5000, pairs ≈ 12.5M - feasible
    # If pw_limit = 5*10^7, n_pw ≈ 8000, pairs ≈ 32M - maybe feasible in 150s
    
    # Let's use pw_limit = 2*10^7 for the sumset approach
    pw_limit2 = 2 * 10**7
    powerful2 = _generate_powerful(pw_limit2)
    powerful2_set = set(powerful2)
    n_pw2 = len(powerful2)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Now count sums for N up to 2 * pw_limit2
    # Use a defaultdict for counting
    sum_count = defaultdict(int)
    max_N2 = 2 * pw_limit2
    
    # For each a, for each b >= a, if a + b <= max_N2:
    #   sum_count[a+b] += 1
    # Optimize: for each a, b ranges from a to max_N2 - a
    # Use bisect to find valid range in powerful2
    
    import bisect
    
    for i, a in enumerate(powerful2):
        if a > max_N2 - a:  # a + a > max_N2, no valid b
            break
        # b must be in [a, max_N2 - a]
        b_min = a
        b_max = max_N2 - a
        # Find indices in powerful2
        lo = bisect.bisect_left(powerful2, b_min)
        hi = bisect.bisect_right(powerful2, b_max)
        for j in range(lo, hi):
            b = powerful2[j]
            s = a + b
            sum_count[s] += 1
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Return best found so far
        if sum_count:
            best_N = max(sum_count, key=sum_count.get)
            best_count = sum_count[best_N]
            pairs = [[a, best_N - a] for a in powerful2 if a <= best_N - a and best_N - a in powerful2_set]
            return {'N': best_N, 'pairs': pairs}
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the N with maximum representations
    best_N = max(sum_count, key=sum_count.get)
    best_count = sum_count[best_N]
    
    # Verify and build pairs
    pairs = []
    for a in powerful2:
        if a > best_N - a:
            break
        b = best_N - a
        if b in powerful2_set:
            pairs.append([a, b])
    
    # If we have time, try to extend to larger N using the larger powerful set
    # For larger N, we can have one number from powerful (up to 10^9) and one from powerful2
    # But this gets complicated. Let's also try a different strategy.
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30 and n_pw2 > 1000:
        # Try to find better N by considering N = a + b where a is from larger set
        # and b is from smaller set, for N in a targeted range
        # Actually, let's try to directly search for N with many representations
        # by checking candidate N values
        
        # Strategy: For a candidate N, count = sum over a in powerful (a <= N/2) of [N-a in powerful_set]
        # Try N values that are sums of many small powerful numbers
        
        # The best_N we found is a good candidate. Let's also try nearby values.
        # And try some larger N values.
        
        # For larger N, use the larger powerful set
        # Count representations for N using powerful (up to 10^9)
        # But this requires checking if N - a is in powerful_set for many a
        
        # Let's try N values in the range [best_N, best_N + 10^6] using the larger set
        # And also try some random larger N values
        
        # Actually, let's just try to improve by checking if we can find a better N
        # in a targeted way. The sumset approach already gives us the best N in [0, 2*pw_limit2].
        
        # Let's verify our answer is correct and return it
        pass
    
    # Final verification
    assert best_N >= 2
    assert len(pairs) == best_count
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
