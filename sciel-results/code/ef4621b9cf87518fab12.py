import time
import math
from collections import deque

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes_all = sieve(5000)
    
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

    best_pair = None
    best_M = float('inf')
    
    # Try small M first
    for M in range(2, 2000):
        if time.time() > deadline - 3:
            break
            
        primes_up_to_M = [p for p in primes_all if p <= M]
        
        smooth_nums = generate_smooth_numbers(TARGET, primes_up_to_M)
        
        # Check for pair
        found = False
        for a in smooth_nums:
            if time.time() > deadline - 3:
                break
            if a <= 0 or a >= TARGET:
                continue
            b = TARGET - a
            if b <= 0:
                continue
            if b in smooth_nums:
                best_pair = [a, b]
                best_M = M
                found = True
                break
        
        if found:
            # Since we iterate M in increasing order, first valid M is optimal
            return best_pair
        
    # Fallback: return best pair found, or a valid pair
    if best_pair is not None:
        return best_pair
    return [1, TARGET]
