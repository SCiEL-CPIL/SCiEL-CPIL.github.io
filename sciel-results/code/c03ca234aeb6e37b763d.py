import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Generate primes up to 10^6 for factoring numbers up to 10^12
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Function to compute phi(n) using the primes list
    def phi(n):
        result = n
        y = n
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    # We'll search in windows. For each window starting at s, compute phi(s), ..., phi(s+199)
    # and check for pairwise distinctness.
    
    # Strategy: 
    # 1. First, do a quick scan over small ranges to find a good baseline.
    # 2. Then, use a segmented approach: for larger s, compute phi for a window of 200 numbers.
    #    To speed up, we can precompute phi for ranges where numbers are smooth, but for large s,
    #    we just compute each phi individually. However, we can optimize by checking collisions
    #    as we go.
    
    # Let's start with a scan from 1 to some limit, then move to larger ranges.
    # The key insight: for large n, phi(n) is roughly n * prod(1-1/p). Collisions are more likely
    # when n has similar prime factorizations. We want to find ranges where phi values are all distinct.
    
    # A good heuristic: start near numbers that are products of distinct small primes (primorials),
    # but actually, we want to AVOID numbers with many small prime factors because they cause
    # phi values to be "compressed". Numbers that are prime or have large prime factors tend to
    # have phi values close to n, which are more likely to be distinct.
    
    # Let's do a systematic search. We'll scan windows of size 200.
    # For efficiency, we'll jump by some step to avoid redundant work.
    
    best_k = 1
    best_s = 1
    
    # First, scan small ranges quickly
    # We'll scan from s=1 up to some point, checking windows of 200
    scan_limit = 100000
    for s in range(1, min(scan_limit, 10**12 - 199)):
        if time.time() > deadline - 5:
            break
        seen = set()
        k = 0
        for i in range(200):
            n = s + i
            if n > 10**12:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
    
    # Now scan larger ranges. We'll use a step to avoid checking every single s.
    # But we need to be careful: the optimal s might be anywhere.
    # Let's scan in chunks, jumping by, say, 1000, and within each chunk, check all 200-length windows.
    # Actually, that's still expensive. Let's think differently.
    
    # Alternative: for a given s, we compute phi(s), phi(s+1), ..., phi(s+199). 
    # The bottleneck is computing phi for 200 numbers. Each phi takes O(sqrt(n)/log(n)) time.
    # For n ~ 10^12, sqrt(n) = 10^6, so each phi takes ~10^6 operations. 200 numbers = 2*10^8 ops.
    # That's too slow for many windows.
    
    # We need a smarter approach. Let's precompute phi for a range using a segmented sieve.
    
    # Segmented phi computation:
    # For a window [L, R], we want phi(n) for all n in [L, R].
    # We can use the fact that phi(n) = n * prod_{p|n}(1-1/p).
    # For each prime p, we can mark all multiples of p in [L, R] and multiply by (1-1/p).
    
    def compute_phi_window(L, R, primes):
        """Compute phi(n) for all n in [L, R] using segmented sieve."""
        window_size = R - L + 1
        phi_vals = list(range(L, R + 1))  # phi_vals[i] will hold phi(L+i)
        
        for p in primes:
            if p > R:
                break
            # Find the first multiple of p in [L, R]
            first = ((L + p - 1) // p) * p
            if first < p * p and first == p:
                # p itself: phi(p) = p-1, which is handled by the general formula
                # Actually, the formula phi(n) = n * prod(1-1/p) works for all n
                pass
            for j in range(first, R + 1, p):
                idx = j - L
                phi_vals[idx] -= phi_vals[idx] // p
        
        return phi_vals
    
    # Now scan windows using this method
    # We'll scan from L=1 up to 10^12, stepping by some amount
    # But we can't scan all the way. Let's be strategic.
    
    # First, let's try some promising ranges:
    # 1. Near primorials
    # 2. Near powers of 2
    # 3. Random samples
    
    # Let's define a list of candidate starting points
    candidates = []
    
    # Small range scan (already done above for s < 100000)
    
    # Add some strategic points
    # Primorials: 2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230]
    for p in primorials:
        if p <= 10**12 - 199:
            candidates.append(p)
            candidates.append(p - 100)
            candidates.append(p + 100)
    
    # Powers of 2
    for i in range(1, 40):
        p = 2**i
        if p <= 10**12 - 199:
            candidates.append(p)
            candidates.append(p - 100)
            candidates.append(p + 100)
    
    # Some random-ish points (deterministic pseudo-random)
    import random
    rng = random.Random(42)
    for _ in range(1000):
        p = rng.randint(1, 10**12 - 199)
        candidates.append(p)
    
    # Also add multiples of large primes
    # For large primes q, numbers near q have phi values that are close to n, so distinctness is likely
    for i, q in enumerate(primes[-100:]):  # last 100 primes < 10^6
        # q is around 10^6, so q^2 is around 10^12
        p = q * q
        if p <= 10**12 - 199:
            candidates.append(p)
            candidates.append(p - 100)
            candidates.append(p + 100)
    
    # Deduplicate and sort
    candidates = sorted(set(c for c in candidates if 1 <= c <= 10**12 - 199))
    
    # Scan these candidates
    for L in candidates:
        if time.time() > deadline - 5:
            break
        R = min(L + 199, 10**12)
        if R - L + 1 < 200:
            continue
        phi_vals = compute_phi_window(L, R, primes)
        seen = set()
        k = 0
        for i, v in enumerate(phi_vals):
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = L
    
    # If we have time, do a denser scan around the best found
    if time.time() < deadline - 10 and best_s > 1:
        # Scan around best_s
        for offset in range(-500, 501):
            if time.time() > deadline - 5:
                break
            L = best_s + offset
            if L < 1 or L > 10**12 - 199:
                continue
            R = min(L + 199, 10**12)
            if R - L + 1 < 200:
                continue
            phi_vals = compute_phi_window(L, R, primes)
            seen = set()
            k = 0
            for i, v in enumerate(phi_vals):
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_s = L
    
    # Final: ensure we return a valid result
    if best_k < 1:
        best_k = 1
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
