import math
import time
import sys

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 256

    # 1. Sieve primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Map prime to bit index
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    # 2. Compute bitmasks for each number 1..N
    masks = [0] * (N + 1)
    for n in range(1, N + 1):
        x = n
        m = 0
        for i, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                m |= (1 << i)
                while x % p == 0:
                    x //= p
        if x > 1:
            if x in prime_to_bit:
                m |= (1 << prime_to_bit[x])
        masks[n] = m
    masks[1] = 0

    # 3. Sort numbers 1..N in descending order
    numbers = list(range(1, N + 1))
    numbers.sort(reverse=True)
    sorted_vals = numbers
    sorted_masks = [masks[n] for n in numbers]
    
    # Precompute suffix sums for a very loose bound if needed, 
    # but we use the tight bound in the loop.
    
    best_sum = 0
    best_set = []
    
    # Increase recursion limit for safety
    sys.setrecursionlimit(3000)
    
    def search(idx, used_mask, curr_sum, curr_set):
        nonlocal best_sum, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: Upper bound
        # Sum of all remaining numbers that don't conflict with used_mask
        bound = curr_sum
        for j in range(idx, N):
            if (used_mask & sorted_masks[j]) == 0:
                bound += sorted_vals[j]
        
        if bound <= best_sum:
            return

        if idx == N:
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_set = list(curr_set)
            return

        val = sorted_vals[idx]
        mask = sorted_masks[idx]
        
        # Option 1: Include val
        if (used_mask & mask) == 0:
            curr_set.append(val)
            search(idx + 1, used_mask | mask, curr_sum + val, curr_set)
            curr_set.pop()
            
        # Option 2: Exclude val
        search(idx + 1, used_mask, curr_sum, curr_set)

    search(0, 0, 0, [])
    
    # 1 is always beneficial and coprime with all
    if 1 not in best_set:
        best_set.append(1)
        best_sum += 1

    result = [int(x) for x in best_set]
    
    # Validation
    if not result:
        return [1]
        
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: Greedy Prime Powers
        fallback = [1]
        for p in primes:
            val = p
            while val * p <= N:
                val *= p
            fallback.append(val)
        return [int(x) for x in fallback]
        
    return result
