"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _pairwise_coprime(vals):
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _validate(A, B):
    if len(A) != 3 or len(B) != 3:
        return 0
    if len(set(A)) != 3 or len(set(B)) != 3:
        return 0
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return 0
    sums = set(a + b for a in A for b in B)
    if not _pairwise_coprime(list(sums)):
        return 0
    return len(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s limit
    
    # Precompute primes up to 256 (max sum is 128+128=256)
    primes = set()
    for i in range(2, 257):
        if _is_prime(i):
            primes.add(i)
    
    # For each a, find all b in [1,128] such that a+b is prime
    valid_b_for_a = {}
    for a in range(1, 129):
        valid_b_for_a[a] = set(b for b in range(1, 129) if (a + b) in primes)
    
    best_result = None
    best_distinct_count = 0
    
    # Try all triples from A
    for a1 in range(1, 129):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 129):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline:
                    break
                
                # Find common b values that work with all three a values
                common_bs = valid_b_for_a[a1].intersection(
                    valid_b_for_a[a2]).intersection(valid_b_for_a[a3])
                
                if len(common_bs) < 3:
                    continue
                
                # Try all triples of b from common_bs
                bs = sorted(common_bs)
                for i in range(len(bs)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(bs)):
                        if time.time() > deadline:
                            break
                        for k in range(j + 1, len(bs)):
                            if time.time() > deadline:
                                break
                            
                            A = [a1, a2, a3]
                            B = [bs[i], bs[j], bs[k]]
                            distinct_count = _validate(A, B)
                            
                            if distinct_count > best_distinct_count:
                                best_distinct_count = distinct_count
                                best_result = (A, B)
    
    if best_result is None:
        # Fallback to the seed solution
        return [[1, 7, 13], [4, 10, 16]]
    
    return best_result
# EVOLVE-BLOCK-END
