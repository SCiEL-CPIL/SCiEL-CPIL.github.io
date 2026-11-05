import time
import random
from itertools import combinations, product

def _prime_masks():
    MAX = 256
    is_prime = [True] * (MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX + 1, i):
                is_prime[j] = False
    
    odd_primes = [p for p in range(3, MAX + 1) if is_prime[p]]
    prime_to_bit = {p: i for i, p in enumerate(odd_primes)}
    
    masks = [0] * (MAX + 1)
    for n in range(1, MAX + 1):
        if n % 2 == 0:
            continue
        mask = 0
        for p in odd_primes:
            if p > n:
                break
            if n % p == 0:
                mask |= (1 << prime_to_bit[p])
        masks[n] = mask
    return masks

def _pairwise_coprime_masks(masks_list):
    n = len(masks_list)
    for i in range(n):
        for j in range(i+1, n):
            if masks_list[i] & masks_list[j]:
                return False
    return True

def _check_valid(A, B, masks):
    """Check if A and B produce pairwise coprime distinct sums. Return count of distinct sums."""
    sums_set = set()
    for a in A:
        for b in B:
            s = a + b
            if s > 256:
                return 0
            sums_set.add(s)
    
    distinct_sums = list(sums_set)
    count = len(distinct_sums)
    
    if count <= 1:
        return count
    
    sum_masks = [masks[s] for s in distinct_sums]
    if _pairwise_coprime_masks(sum_masks):
        return count
    return 0

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    masks = _prime_masks()
    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Phase 1: Deterministic search on small anchor sets
    # Check all combinations of small odds and evens
    small_odds = odds[:12]  # First 12 odds: 1,3,5,...,23
    small_evens = evens[:12]  # First 12 evens: 2,4,6,...,24
    
    start_time = time.time()
    
    # Check all small A and B combinations
    for A in combinations(small_odds, 3):
        if time.time() > deadline - 5:
            break
        for B in combinations(small_evens, 3):
            count = _check_valid(A, B, masks)
            if count > best_count:
                best_count = count
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return [best_A, best_B]
    
    # Phase 2: Random search with better heuristics
    # Bias toward sets with larger gaps to encourage distinct sums
    while time.time() < deadline - 5:
        # Generate A with some randomness but biased toward spread
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        
        # Quick check: if any sum exceeds 256, skip
        max_sum = max(A) + max(B)
        if max_sum > 256:
            continue
        
        count = _check_valid(A, B, masks)
        if count > best_count:
            best_count = count
            best_A = A
            best_B = B
            if best_count == 9:
                return [best_A, best_B]
    
    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]
