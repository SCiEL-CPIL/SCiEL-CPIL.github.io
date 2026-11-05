import time
from math import isqrt
from collections import Counter

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > temp:
            break
        if temp % (p * p) == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    powerful.add(1)
    max_b = int(limit ** (1/3)) + 2
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for b in range(1, max_b + 1):
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
            while t % p == 0:
                t //= p
        if not is_sf:
            continue
        if t > 1:
            r = isqrt(t)
            if r * r == t:
                is_sf = False
        if not is_sf:
            continue
            
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    # Generate powerful numbers up to a limit that allows O(K^2) counting within time
    # Based on knowledge: 80M limit -> ~23000 powerful numbers
    # 23000^2 / 2 = 265M operations. In Python this might be too slow (~50-100s+).
    # Let's aim for a limit that gives ~10000-15000 numbers.
    # 20M -> ~7000 numbers. 7000^2/2 = 24.5M ops. This is fast (~1-2s).
    # 50M -> ~15000 numbers? 15000^2/2 = 112M ops. This is ~10-20s in Python.
    # Let's try 50M first. If time permits, we can increase.
    # Actually, the insight says 80M gave 5.933 vs 4.133.
    # Current program uses 10^10 limit but only counts up to deadline.
    # The current program generates powerful numbers up to 10^10, which is ~5669 numbers?
    # Wait, the comment in current program says "K approx 5669" for 10^10.
    # Let's check: number of powerful numbers up to 10^10.
    # Formula is roughly 2.16 * x^(1/2) ... no, it's x^(1/2) * something?
    # Actually, count of powerful numbers up to x is ~ x^(1/2) / (2 * pi) * something?
    # No, it's known to be ~ x^(1/2).
    # For x=10^10, sqrt(x)=10^5. But the count is much smaller.
    # Let's just generate and see.
    
    pow_limit = 5 * 10**7
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    # Check time
    if time.time() > deadline - 60:
        # If generation took too long, we are in trouble, but it should be fast.
        pass

    # Count sums
    # K ~ 15000 for 50M? Let's assume.
    # K^2 / 2 operations.
    sum_counts = Counter()
    pow_list = powerful
    counts = sum_counts
    
    # Optimization: Use local variables for speed
    # We can also use numpy if available, but standard library is safer for this logic.
    # However, a pure Python double loop over 15000 items is 112.5M iterations.
    # Each iteration is simple addition and dict update.
    # This might take 20-30 seconds.
    
    # Let's try to optimize the inner loop.
    # Alternative: sort list, use two pointers? No, we need all sums.
    
    # Let's try a slightly smaller limit to ensure speed, then maybe expand if time remains?
    # No, we can't easily expand.
    # Let's stick with 50M.
    
    for i in range(K):
        a = pow_list[i]
        for j in range(i, K):
            b = pow_list[j]
            s = a + b
            counts[s] += 1
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    if not counts:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Extract pairs
    pow_set = set(pow_list)
    pairs = []
    for a in pow_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in pow_set and a <= b:
            pairs.append([a, b])
            
    # Verify
    verified_pairs = []
    for p in pairs:
        if _is_powerful(p[0]) and _is_powerful(p[1]):
            verified_pairs.append(p)
            
    if not verified_pairs:
        verified_pairs = [[1, 4]]
        best_N = 5
        
    return {'N': best_N, 'pairs': verified_pairs}
