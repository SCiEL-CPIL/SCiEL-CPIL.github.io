import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Precompute prime factor masks for numbers 1..256
    # Primes up to 256
    primes = []
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, 257):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Map prime to bit index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    pf_mask = [0] * 257
    for n in range(1, 257):
        mask = 0
        for p in primes:
            if n % p == 0:
                mask |= (1 << prime_to_idx[p])
        pf_mask[n] = mask

    # Candidates: 1 to 128
    # We prioritize small numbers as they are more likely to be coprime to others?
    # Actually, primes are best.
    # Let's just try 1..60 first.
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 0
    
    def check(A, B):
        sums = []
        for a in A:
            for b in B:
                s = a + b
                sums.append(s)
        
        # Check distinct
        if len(set(sums)) != 9:
            return 0
            
        # Check pairwise coprime using masks
        for i in range(9):
            for j in range(i+1, 9):
                if pf_mask[sums[i]] & pf_mask[sums[j]]:
                    return 0
        return 9

    # Initial check
    s = check(best_A, best_B)
    if s > best_score:
        best_score = s
        if best_score == 9:
            return [best_A, best_B]

    # Search
    # Try small ranges first
    for limit in [30, 50, 80, 128]:
        candidates = list(range(1, limit + 1))
        
        # Optimization: If A is fixed, we can precompute something?
        # Just iterate.
        # To save time, we can break early if we find 9.
        
        # We can also restrict B candidates based on A?
        # For now, simple nested loops.
        
        # To avoid O(N^6) explosion, we use the fact that N is small (<=128).
        # But 128C3 is ~350k. 350k^2 is 1e11. Too slow.
        
        # We must prune.
        # Observation: If A has 3 numbers, and B has 3 numbers.
        # We can iterate A, and for each A, iterate B.
        # But we can filter B candidates?
        
        # Alternative:
        # Iterate over all pairs of (A, B) is too slow.
        # Let's try to find a solution with A, B from 1..30.
        
        if time.time() > deadline:
            break
            
        for A in combinations(candidates, 3):
            if time.time() > deadline:
                break
            A = list(A)
            # Quick check on A? No, A elements don't need to be coprime.
            
            for B in combinations(candidates, 3):
                if time.time() > deadline:
                    break
                B = list(B)
                s = check(A, B)
                if s > best_score:
                    best_score = s
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return [best_A, best_B]
                        
    return [best_A, best_B]
