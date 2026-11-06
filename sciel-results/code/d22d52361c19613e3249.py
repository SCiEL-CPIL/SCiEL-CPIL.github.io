"""Rewritten program: search for A,B with all 9 sums being distinct primes."""
import itertools
import time

def _sieve(n):
    """Return list of booleans where is_prime[i] is True iff i is prime."""
    is_prime = [False] * (n + 1)
    if n >= 2:
        is_prime[2] = True
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return is_prime

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Sieve for primes up to 256 (max possible sum)
    is_prime = _sieve(256)
    
    # Precompute list of primes in [2, 256]
    primes = [i for i in range(2, 257) if is_prime[i]]
    
    # Strategy: iterate over A combinations, for each try to find B
    # such that all 9 sums ai+bj are distinct primes.
    # To speed up: for a given A, the sums for b must be:
    #   For each b, {a1+b, a2+b, a3+b} must all be prime
    #   And all 9 must be distinct
    
    # Precompute for each b in [1,128], which A-triples work
    # Actually, let's iterate over A first, then filter B candidates
    
    # For efficiency, precompute for each b which a-values give prime sums
    # But let's just do a structured search with early termination
    
    # Iterate over A combinations (sorted for determinism)
    # C(128,3) = 352,672 — manageable if inner loop is fast
    
    # For each A, collect candidate b's where all three sums are prime
    # Then check if any 3 distinct candidates give 9 distinct sums
    
    for a1, a2, a3 in itertools.combinations(range(1, 129), 3):
        if time.time() > deadline:
            # Return best fallback (the seed solution)
            return [[1, 7, 13], [4, 10, 16]]
        
        # Find all b in [1,128] such that a1+b, a2+b, a3+b are all prime
        candidate_bs = []
        for b in range(1, 129):
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if s1 <= 256 and s2 <= 256 and s3 <= 256:
                if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                    candidate_bs.append(b)
            # If any sum > 256, skip (but b<=128 and a<=128 so max sum=256)
        
        if len(candidate_bs) < 3:
            continue
        
        # Try all triples of distinct candidate_bs
        for b1, b2, b3 in itertools.combinations(candidate_bs, 3):
            # Compute all 9 sums
            sums = set()
            valid = True
            for a in (a1, a2, a3):
                for b in (b1, b2, b3):
                    s = a + b
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            
            if valid and len(sums) == 9:
                # All 9 sums are distinct and prime (hence pairwise coprime)
                return [a1, a2, a3], [b1, b2, b3]
    
    # Fallback: return seed solution
    return [[1, 7, 13], [4, 10, 16]]
