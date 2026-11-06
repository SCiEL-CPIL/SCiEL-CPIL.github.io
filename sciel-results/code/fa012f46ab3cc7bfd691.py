"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute smallest prime factor for fast factorization
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def evaluate(arr):
        """Return the number of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)
    
    def get_prime_set(arr):
        """Return the set of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return primes
    
    # Strategy: Graph-based clique search
    # Build a graph where vertices are integers 1..10000
    # Edge (u,v) exists if u+v is "smooth" (has few prime factors)
    # Find a clique of size 12 (mutually compatible numbers)
    
    # First, let's identify "smooth" sums - sums with few distinct prime factors
    # We want to find 12 numbers where all pairwise sums are smooth
    
    # Precompute which numbers have few prime factors
    # A number is k-smooth if all its prime factors are <= k
    # But we want to minimize the UNION of prime factors across all sums
    
    # Better approach: Try different prime sets P and find numbers where all pairwise sums are P-smooth
    # P-smooth means all prime factors are in P
    
    # Let's try small prime sets:
    # P = {2}: sums must be powers of 2 - very restrictive
    # P = {2, 3}: sums must be of form 2^a * 3^b
    # P = {2, 3, 5}: sums must be of form 2^a * 3^b * 5^c
    # etc.
    
    # For each candidate prime set, build a graph and find a clique of size 12
    
    # Precompute: for each number 1..10000, check if it's P-smooth for various P
    # But that's expensive. Instead, let's use a heuristic:
    
    # Start with numbers that are multiples of small numbers (likely to have smooth sums)
    # Then use local search to refine
    
    # Initial candidates: multiples of 6 (2*3)
    best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    best_score = evaluate(best)
    best_primes = get_prime_set(best)
    
    # Also try other structured sets
    alternative_sets = [
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],  # powers of 2
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],  # multiples of 3
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],  # multiples of 4
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],  # multiples of 5
    ]
    
    for alt in alternative_sets:
        score = evaluate(alt)
        if score < best_score:
            best = alt
            best_score = score
            best_primes = get_prime_set(best)
    
    # Local search with candidate pool
    random.seed(42)
    
    # Generate candidate values: focus on values that are multiples of small numbers
    candidate_pool = []
    for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
        for k in range(1, 10000 // m + 1):
            val = k * m
            if 1 <= val <= 10000:
                candidate_pool.append(val)
    # Add some random values
    for _ in range(5000):
        candidate_pool.append(random.randint(1, 10000))
    candidate_pool = list(set(candidate_pool))
    random.shuffle(candidate_pool)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all possible replacements
        for i in range(12):
            old_val = best[i]
            # Try replacing with values from the candidate pool
            for new_val in candidate_pool:
                if time.time() > deadline:
                    break
                if new_val == old_val:
                    continue
                if new_val in best:
                    continue
                candidate = best.copy()
                candidate[i] = new_val
                score = evaluate(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    best_primes = get_prime_set(best)
                    improved = True
                    break
    
    return best
# EVOLVE-BLOCK-END
