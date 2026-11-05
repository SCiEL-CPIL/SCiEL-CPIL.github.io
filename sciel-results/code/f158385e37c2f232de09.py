import itertools
from math import gcd

def construct():
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    prime_factors = {i: get_prime_factors(i) for i in range(2, 257)}
    
    def is_pairwise_coprime(nums):
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if prime_factors[nums[i]] & prime_factors[nums[j]]:
                    return False
        return True
    
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    sums = [a + b for a in best_A for b in best_B]
    ds = list(set(sums))
    if is_pairwise_coprime(ds):
        best_count = len(ds)
    
    # Focused search with smaller candidate sets
    # Use values 1..60 for A and B
    a_candidates = list(range(1, 61))
    b_candidates = list(range(1, 61))
    
    # Sort by prime factor count
    a_candidates.sort(key=lambda x: len(prime_factors[x]) if x >= 2 else 0)
    b_candidates.sort(key=lambda x: len(prime_factors[x]) if x >= 2 else 0)
    
    a_candidates = a_candidates[:50]
    b_candidates = b_candidates[:50]
    
    for A in itertools.combinations(a_candidates, 3):
        A = list(A)
        # Quick check: can we even get 9 distinct sums?
        # a1+b1 < a1+b2 < a1+b3, etc. Need all 9 distinct.
        # This requires b1,b2,b3 distinct (which they are) and
        # a_i + b_j all distinct, which requires a_i distinct (they are)
        # and no a_i + b_j = a_k + b_l for i!=k or j!=l
        # i.e., a_i - a_k = b_l - b_j for no i!=k, j!=l
        
        valid_b = []
        for b in b_candidates:
            sums_b = [a + b for a in A]
            if len(set(sums_b)) != 3:
                continue
            if is_pairwise_coprime(sums_b):
                valid_b.append(b)
        
        if len(valid_b) < 3:
            continue
        
        # For each pair of valid b's, check if they're compatible
        # Then find triples
        n_valid = len(valid_b)
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                for k in range(j + 1, n_valid):
                    B = [valid_b[i], valid_b[j], valid_b[k]]
                    sums = [a + b for a in A for b in B]
                    distinct_sums = set(sums)
                    if len(distinct_sums) <= best_count:
                        continue
                    ds = list(distinct_sums)
                    if is_pairwise_coprime(ds):
                        best_count = len(ds)
                        best_A = A
                        best_B = B
                        if best_count == 9:
                            return [best_A, best_B]
    
    return [best_A, best_B]
