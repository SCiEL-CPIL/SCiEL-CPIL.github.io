import time
from math import isqrt
from collections import defaultdict, Counter
import numpy as np

def generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    pows = set([1])
    # Powerful numbers are of the form a^2 * b^3 where b is square-free.
    max_b = int(limit ** (1/3)) + 1
    
    # Precompute primes up to sqrt(max_b) for square-free check
    # max_b is ~2154 for 10^10, sqrt is ~46
    primes = []
    is_prime = [True] * (int(max_b**0.5) + 1)
    for i in range(2, int(max_b**0.5) + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, int(max_b**0.5) + 1, i):
                is_prime[j] = False

    def is_square_free(n):
        if n <= 1:
            return True
        for p in primes:
            if p * p > n:
                break
            if n % (p * p) == 0:
                return False
        return True

    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        
        b_cubed = b ** 3
        if b_cubed > limit:
            break
            
        max_a = isqrt(limit // b_cubed)
        for a in range(1, max_a + 1):
            n = a * a * b_cubed
            if n <= limit:
                pows.add(n)
            else:
                break
                
    return sorted(pows)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Limit chosen to balance K and computation time
    # K ~ 34000 for 10^10
    limit = 10**10
    powerful_list = generate_powerful(limit)
    powerful_set = set(powerful_list)
    num_powerful = len(powerful_list)
    
    if num_powerful == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    arr = np.array(powerful_list, dtype=np.int64)
    
    rep_count = defaultdict(int)
    
    # Iterate over each powerful number as the first element 'a'
    for i in range(num_powerful):
        a = arr[i]
        # Sums with all b >= a
        sums = a + arr[i:]
        
        # Convert to list for Counter
        # This is the bottleneck, but C-optimized
        lst = sums.tolist()
        c = Counter(lst)
        
        # Update global count
        for key, val in c.items():
            rep_count[key] += val
            
        # Time check
        if i % 1000 == 0 and time.time() > deadline:
            break
            
    if not rep_count:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(rep_count.keys(), key=lambda n: rep_count[n])
    
    # Collect pairs for best_N
    pairs = []
    for a in powerful_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
