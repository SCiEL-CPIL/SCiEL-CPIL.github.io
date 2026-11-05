import itertools
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    
    # We focus on small primes for high density
    # Let's consider the first 20 primes
    candidate_primes = primes[:20]
    
    best_L = 0
    best_classes = []
    
    # Generate all subsets of candidate_primes
    # We can prune subsets where sum of 1/p > 1
    # We also want to maximize coverage, so we prefer subsets with high density (low product of (1-1/p))
    
    # To be efficient, we can iterate over subset sizes or just all subsets
    # 2^20 is 1 million, which is fine.
    
    for r in range(len(candidate_primes) + 1):
        for subset in itertools.combinations(candidate_primes, r):
            # Check budget
            budget_sum = sum(Fraction(1, p) for p in subset)
            if budget_sum > 1:
                continue
            
            # Optional: Skip if density is too low to beat current best?
            # Density = 1 - product(1 - 1/p)
            # If density * L_max < best_L, skip?
            # L_max is unknown, but we can estimate.
            
            # Calculate all residue combinations
            ranges = [range(p) for p in subset]
            
            # If the product of primes is too large, we might have too many combinations
            # But for small primes, it's manageable.
            # Max product for valid subset:
            # 2*3*7*43 = 1806
            # 2*3*11*17*59 = 68454 (if valid)
            # Let's check if 2,3,11,17,59 is valid.
            # 1/2+1/3+1/11+1/17+1/59 = 0.5+0.333+0.0909+0.0588+0.0169 = 0.9996. Valid.
            # Product 68454. 68k combinations is fine.
            
            for residues in itertools.product(*ranges):
                # Check coverage
                # We simulate x = 1, 2, ...
                # We stop when we find an uncovered x
                # We also stop if time is running out
                
                if time.time() > deadline - 10:
                    return best_classes
                
                L = 0
                covered = True
                x = 1
                while covered:
                    # Check if x is covered by any (p, r)
                    is_covered = False
                    for p, r_val in zip(subset, residues):
                        if x % p == r_val:
                            is_covered = True
                            break
                    if is_covered:
                        x += 1
                    else:
                        covered = False
                
                # The largest L is x - 1
                if x - 1 > best_L:
                    best_L = x - 1
                    best_classes = [[p, r_val] for p, r_val in zip(subset, residues)]
                    
                    # If we reach a very high L, we might want to break early?
                    # Theoretical max L for density d is roughly 1/(1-d) * log(1/(1-d))?
                    # Anyway, we keep searching for better subsets.
    
    return best_classes
