"""Optimized search for pairwise-coprime sumsets."""
import math
import time
import random
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    
    best_cardinality = 0
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    def check_pairwise_coprime(values):
        """Check if all distinct values in the set are pairwise coprime."""
        distinct_vals = list(set(values))
        n = len(distinct_vals)
        for i in range(n):
            vi = distinct_vals[i]
            for j in range(i + 1, n):
                if gcd(vi, distinct_vals[j]) != 1:
                    return False
        return True
    
    def try_pair(A, B):
        nonlocal best_cardinality, best_result
        A_sorted = sorted(A)
        B_sorted = sorted(B)
        # Verify distinct and in range
        if len(set(A_sorted)) != 3 or len(set(B_sorted)) != 3:
            return False
        if any(x < 1 or x > 128 for x in A_sorted) or any(x < 1 or x > 128 for x in B_sorted):
            return False
        
        sums = [a + b for a in A_sorted for b in B_sorted]
        distinct_sums = set(sums)
        
        if len(distinct_sums) <= best_cardinality:
            return False
        
        if check_pairwise_coprime(list(distinct_sums)):
            best_cardinality = len(distinct_sums)
            best_result = [A_sorted[:], B_sorted[:]]
            return True
        return False
    
    # Precompute primes up to 256 for faster coprimality checks
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(256)
    
    # Strategy 1: Try structured patterns with arithmetic progressions
    for a in range(1, 80):
        if time.time() > deadline:
            return best_result
        for d in range(1, 30):
            A = [a, a + d, a + 2 * d]
            if A[2] > 128:
                break
            for b in range(1, 80):
                for e in range(1, 30):
                    B = [b, b + e, b + 2 * e]
                    if B[2] > 128:
                        break
                    try_pair(A, B)
                if time.time() > deadline:
                    return best_result
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 2: Random search with bias toward numbers with few small prime factors
    random.seed(42)
    
    # Generate candidate numbers weighted toward those likely to produce coprime sums
    # Numbers that are prime or products of large primes are better
    def weight_number(n):
        """Higher weight for numbers with fewer small prime factors."""
        if n < 2:
            return 1
        score = 1.0
        for p in [2, 3, 5, 7, 11, 13]:
            if n % p == 0:
                score *= 0.3
        if is_prime[n]:
            score *= 2.0
        return score
    
    # Create weighted list of candidates
    candidates = list(range(1, 129))
    weights = [weight_number(n) for n in candidates]
    
    def weighted_sample(k):
        """Sample k distinct elements with replacement from weighted candidates."""
        chosen = set()
        attempts = 0
        while len(chosen) < k and attempts < 1000:
            # Weighted random selection
            total_w = sum(w for i, w in enumerate(weights) if i+1 not in chosen)
            if total_w <= 0:
                break
            r = random.random() * total_w
            cum = 0
            for i, w in enumerate(weights):
                if i+1 in chosen:
                    continue
                cum += w
                if cum >= r:
                    chosen.add(i+1)
                    break
            attempts += 1
        if len(chosen) < k:
            return None
        return sorted(list(chosen))
    
    # Strategy 3: Random search with weighted sampling
    for _ in range(200000):
        if time.time() > deadline:
            return best_result
        A = weighted_sample(3)
        B = weighted_sample(3)
        if A is None or B is None:
            continue
        try_pair(A, B)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 4: Local search around best solution
    if best_result:
        A_best = best_result[0][:]
        B_best = best_result[1][:]
        
        # Try perturbing elements
        for _ in range(50000):
            if time.time() > deadline:
                return best_result
            
            # Perturb A
            A_new = A_best[:]
            idx = random.randint(0, 2)
            delta = random.randint(-5, 5)
            A_new[idx] = max(1, min(128, A_new[idx] + delta))
            if len(set(A_new)) == 3:
                try_pair(A_new, B_best)
            
            # Perturb B
            B_new = B_best[:]
            idx = random.randint(0, 2)
            delta = random.randint(-5, 5)
            B_new[idx] = max(1, min(128, B_new[idx] + delta))
            if len(set(B_new)) == 3:
                try_pair(A_best, B_new)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 5: Targeted search - try to get 9 distinct sums
    # For 9 distinct sums, we need all a_i + b_j to be distinct
    # This happens when the differences in A and B don't create collisions
    # Try sets with large, distinct gaps
    
    # Generate candidate sets with distinct gaps
    def gen_distinct_gap_set():
        """Generate a set with 3 elements having distinct, large gaps."""
        a0 = random.randint(1, 50)
        d1 = random.randint(10, 40)
        d2 = random.randint(10, 40)
        while d2 == d1:
            d2 = random.randint(10, 40)
        A = [a0, a0 + d1, a0 + d1 + d2]
        if A[2] > 128:
            return None
        if len(set(A)) != 3:
            return None
        return sorted(A)
    
    for _ in range(100000):
        if time.time() > deadline:
            return best_result
        A = gen_distinct_gap_set()
        B = gen_distinct_gap_set()
        if A is None or B is None:
            continue
        try_pair(A, B)
    
    return best_result
