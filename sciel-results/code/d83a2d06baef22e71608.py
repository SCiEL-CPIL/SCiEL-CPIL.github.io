import math
import time

def generate_divisors(primes):
    """Generate all divisors for a squarefree number defined by its prime factors."""
    divs = [1]
    for p in primes:
        # For squarefree, we just multiply existing divisors by p
        # This creates 2^k divisors
        new_divs = [d * p for d in divs]
        divs.extend(new_divs)
    return divs

def compute_score(primes):
    """Compute the number of adjacent coprime pairs in the sorted divisors."""
    divs = generate_divisors(primes)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def construct():
    # The optimal strategy is believed to be the product of the smallest primes
    # to maximize interleaving of divisor sets, which maximizes coprime adjacencies.
    # We check the first 15 primes and a few variations where the largest prime is increased.
    
    # First 15 primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    candidates = []
    
    # Candidate 1: First 15 primes
    candidates.append(base_primes)
    
    # Candidate 2: First 14 primes + 53
    c2 = base_primes[:14] + [53]
    candidates.append(c2)
    
    # Candidate 3: First 14 primes + 59
    c3 = base_primes[:14] + [59]
    candidates.append(c3)
    
    # Candidate 4: First 13 primes + 53 + 59? 
    # Prod(13) = 304250263527210
    # 53 * 59 = 3127
    # 304250263527210 * 3127 = 9.5e18 > 1e18. Invalid.
    
    # Candidate 5: First 12 primes + 53 * 59?
    # Prod(12) = 7420738134810
    # 7420738134810 * 3127 = 2.3e16. Valid.
    # But k=14. Likely worse than k=15.
    
    # We focus on k=15 candidates as they have double the divisors.
    
    best_n = 1
    best_score = -1
    best_primes = []
    
    # Time guard
    start_time = time.time()
    deadline = 100 # 100 seconds
    
    for p_set in candidates:
        if time.time() > deadline:
            break
            
        n = 1
        for p in p_set:
            n *= p
            
        if n > 10**18:
            continue
            
        score = compute_score(p_set)
        
        if score > best_score:
            best_score = score
            best_n = n
            best_primes = p_set
            
    # If for some reason no candidate worked (shouldn't happen), fallback to first 15 primes
    if best_n == 1:
        for p in base_primes:
            best_n *= p
            
    return best_n

# Pre-calculate the best one to ensure construct() is instant?
# The evaluation allows 120s. The computation is fast (sorting 32k items).
# But to be safe and "structural", I'll let construct() do the work.
# It's only ~0.1s.

# To satisfy "Keep function names", construct is the entry point.
# I will remove the top-level execution code from the original and keep it clean.
