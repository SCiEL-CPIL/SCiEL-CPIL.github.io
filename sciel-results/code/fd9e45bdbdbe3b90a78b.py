import time
from math import isqrt

def _is_square_free_proper(n):
    """Proper square-free check using trial division up to cbrt(n)."""
    if n < 2:
        return True
    # Small primes for quick elimination
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n == 1:
        return True
    # Check if remaining n is a perfect square
    r = isqrt(n)
    if r * r == n:
        return False
    # Now n has no prime factor <= 97 with multiplicity >= 2
    # n is either 1, prime, product of two primes > 97, or p^k for p > 97
    # If n = p^2 for some prime p > 97, we caught it above
    # If n = p^3 for some prime p > 97, then p^3 > 97^3 = 912673
    # For n > 912673, we need to check if n is a perfect cube
    # But actually, for our purposes (b <= ~2*10^6), after removing primes <= 97,
    # the remaining n is small enough. Let's do a more careful check.
    # Trial divide by odd numbers up to cbrt(n)
    i = 99
    while i * i * i <= n:
        if n % (i * i) == 0:
            return False
        while n % i == 0:
            n //= i
        i += 2
    # Final check: is n a perfect square?
    r = isqrt(n)
    if r * r == n:
        return False
    return True


def _generate_powerful(limit):
    """Generate all powerful numbers <= limit.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    powerful = set([1])
    
    # Max b: b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    
    # For each square-free b, a^2 * b^3 <= limit => a <= sqrt(limit / b^3)
    for b in range(1, max_b + 1):
        if not _is_square_free_proper(b):
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer before 120s limit
    
    # Generate powerful numbers up to a reasonable limit
    # We need N <= 10^18, so powerful numbers up to 10^18
    # But generating all powerful numbers up to 10^18 is too many
    # Instead, generate up to 10^12 or so, and use random sampling for candidates
    
    # Let's generate powerful numbers up to 10^11
    # This should give us a good set to work with
    limit_pw = 10**11
    
    # Generate powerful numbers
    powerful = _generate_powerful(limit_pw)
    powerful_set = set(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 30:
        # Not enough time, use a smaller approach
        limit_pw = 10**9
        powerful = _generate_powerful(limit_pw)
        powerful_set = set(powerful)
    
    # Strategy: Find N with many representations
    # N = a + b where a, b are powerful
    # To find N with many reps, we want many pairs (a, b) from our powerful set that sum to the same N
    
    # Approach: Use a dictionary to count how many ways each sum can be formed
    # But this is O(|powerful|^2) which is too slow for large sets
    
    # Better approach: Random sampling of candidates
    # Pick random powerful numbers a, compute b = N - a for random N, check if b is powerful
    
    # Alternative: For each powerful number a, for each powerful number b, add a+b to a counter
    # But this is O(n^2)
    
    # Let's use a smarter approach:
    # 1. Generate powerful numbers up to limit_pw
    # 2. For a set of candidate N values, count representations
    # 3. Use random sampling to find good N values
    
    # Candidate N values: sums of two powerful numbers
    # Let's sample some sums and count their frequencies
    
    # First, let's try a direct approach for smaller sets
    n_pw = len(powerful)
    
    if n_pw < 500000:
        # Use dictionary to count sums
        sum_count = {}
        for i in range(n_pw):
            a = powerful[i]
            for j in range(i, n_pw):
                b = powerful[j]
                s = a + b
                if s > 10**18:
                    break
                sum_count[s] = sum_count.get(s, 0) + 1
        
        # Find the N with maximum count
        if sum_count:
            best_n = max(sum_count, key=sum_count.get)
            best_count = sum_count[best_n]
            
            # Verify and collect pairs
            pairs = []
            for a in powerful:
                if 2 * a > best_n:
                    break
                b = best_n - a
                if b in powerful_set:
                    pairs.append([a, b])
            
            if pairs:
                return {'N': best_n, 'pairs': pairs}
    
    # For larger sets, use random sampling
    import random
    
    best_n = None
    best_count = 0
    best_pairs = []
    
    # Try random sums of powerful numbers as candidates
    for _ in range(10000):
        if time.time() > deadline - 5:
            break
        
        # Pick two random powerful numbers
        idx1 = random.randint(0, n_pw - 1)
        idx2 = random.randint(0, n_pw - 1)
        a = powerful[idx1]
        b = powerful[idx2]
        n_candidate = a + b
        
        if n_candidate > 10**18:
            continue
        
        # Count representations of n_candidate
        count = 0
        pairs = []
        for pw in powerful:
            if 2 * pw > n_candidate:
                break
            complement = n_candidate - pw
            if complement in powerful_set:
                count += 1
                pairs.append([pw, complement])
        
        if count > best_count:
            best_count = count
            best_n = n_candidate
            best_pairs = pairs
    
    # Also try some structured candidates: multiples of small powerful numbers
    for mult in range(1, 100):
        if time.time() > deadline - 5:
            break
        for a in powerful[:1000]:
            n_candidate = mult * a
            if n_candidate > 10**18:
                break
            count = 0
            pairs = []
            for pw in powerful:
                if 2 * pw > n_candidate:
                    break
                complement = n_candidate - pw
                if complement in powerful_set:
                    count += 1
                    pairs.append([pw, complement])
            if count > best_count:
                best_count = count
                best_n = n_candidate
                best_pairs = pairs
    
    if best_n is None:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_n, 'pairs': best_pairs}
