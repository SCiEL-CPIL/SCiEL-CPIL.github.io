import time
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    TARGET = 123457
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Sieve primes up to a reasonable limit for binary search
    # The max prime factor needed is at most TARGET/2, but likely much smaller.
    # Let's sieve up to 10000 to be safe for binary search range.
    primes_all = sieve(10000)
    
    def generate_smooth_numbers(limit, primes):
        smooth = set([1])
        queue = deque([1])
        while queue:
            x = queue.popleft()
            for p in primes:
                nx = x * p
                if nx > limit:
                    break
                if nx not in smooth:
                    smooth.add(nx)
                    queue.append(nx)
        return smooth

    def find_pair_for_M(M):
        if time.time() > deadline:
            return None, None
        primes_up_to_M = [p for p in primes_all if p <= M]
        if not primes_up_to_M:
            # Only 1 is smooth, but 1+1=2 != TARGET
            return None, None
            
        smooth_nums = generate_smooth_numbers(TARGET, primes_up_to_M)
        
        for a in smooth_nums:
            if a <= 0 or a >= TARGET:
                continue
            b = TARGET - a
            if b in smooth_nums:
                return a, b
        return None, None

    # Binary search for the smallest M
    # We know M=11 works from the current score context, so the answer is <= 11.
    # But let's do a proper binary search from 2 to 10000.
    lo = 2
    hi = 10000
    best_M = None
    best_pair = None
    
    # First, check if a solution exists in our range
    # We can start by checking a high M to ensure we have a valid pair to return if binary search fails or times out
    # But binary search is fast.
    
    while lo <= hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        a, b = find_pair_for_M(mid)
        if a is not None:
            best_M = mid
            best_pair = (a, b)
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_pair is not None:
        return [best_pair[0], best_pair[1]]
        
    # Fallback if binary search didn't find a pair (unlikely)
    # Try linear search from 2 upwards as a fallback
    for M in range(2, 1000):
        if time.time() > deadline - 2:
            break
        a, b = find_pair_for_M(M)
        if a is not None:
            return [a, b]
            
    # Absolute fallback
    return [1, TARGET]
