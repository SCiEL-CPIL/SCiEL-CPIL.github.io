import heapq
import math
import time

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy: Use a priority queue (Dijkstra-like) to generate integers in ascending 
    order of their largest prime factor (LPF). The first pair of consecutive integers 
    (n, n+1) found in the valid range will have the minimal possible max LPF.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Generate primes up to a sufficient limit.
    # The LPF of any number <= 1,000,000 is <= 1,000,000.
    # However, we expect the optimal LPF to be much smaller.
    # Let's generate primes up to 100,000 to be safe, but we can stop early.
    # Actually, we just need primes to multiply.
    # We'll generate primes up to 1000 first, and extend if needed? 
    # No, let's just generate primes up to 10000. 
    # If the optimal LPF is > 10000, the search space is huge. 
    # But typically smooth numbers appear early.
    
    limit_primes = 10000
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit_primes)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit_primes + 1) if is_prime[i]]
    
    # Map from prime to its index for quick lookup? Not strictly needed if we iterate.
    # We need to iterate primes p <= lpf(x).
    
    # Priority Queue: (lpf, number)
    # We start with all primes.
    # Note: We should also consider 1? 
    # If we start with primes, we generate composites.
    # 2, 3, 5...
    # From 2 (lpf=2), generate 2*2=4 (lpf=2).
    # From 4 (lpf=2), generate 4*2=8 (lpf=2).
    # From 3 (lpf=3), generate 3*2=6 (lpf=3), 3*3=9 (lpf=3).
    # This covers all integers >= 2.
    
    heap = []
    for p in primes:
        # We only care about numbers up to N_MAX
        if p <= N_MAX:
            heapq.heappush(heap, (p, p))
            
    # To avoid duplicates, we track visited numbers.
    # However, the generation rule (p <= lpf(x)) ensures unique generation if we are careful.
    # Let's use a set to be safe and efficient for membership testing.
    visited = set()
    
    # We need to check if current number - 1 was visited.
    # We only care about numbers in [N_MIN, N_MAX].
    # But we must pop numbers in order.
    
    # Optimization: We can skip pushing numbers > N_MAX.
    
    while heap:
        if time.time() > deadline:
            # Fallback: Return a reasonable default if timeout
            # This shouldn't happen if the optimal LPF is small.
            break
            
        lpf, num = heapq.heappop(heap)
        
        if num in visited:
            continue
        visited.add(num)
        
        # Check if num is in valid range and num-1 is in valid range and visited
        if num > N_MIN and num <= N_MAX:
            if (num - 1) in visited:
                # We found a pair (num-1, num)
                # The max LPF is lpf (since lpf(num-1) <= lpf(num))
                # We return num-1 as n
                return num - 1
        
        # Generate children
        # Multiply num by primes p such that p <= lpf
        # And num * p <= N_MAX
        for p in primes:
            if p > lpf:
                break
            new_num = num * p
            if new_num > N_MAX:
                break
            # New LPF is max(lpf, p) = lpf
            heapq.heappush(heap, (lpf, new_num))
            
    # Fallback if no pair found (should not happen for this range)
    # Perform a linear scan with trial division as a backup
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lpf = 1
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                lpf = p
                while n % p == 0:
                    n //= p
        if n > 1:
            lpf = n
        return lpf
        
    best_n = N_MIN
    best_val = float('inf')
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline - 1:
            break
        val = max(largest_prime_factor(n), largest_prime_factor(n+1))
        if val < best_val:
            best_val = val
            best_n = n
    return best_n
