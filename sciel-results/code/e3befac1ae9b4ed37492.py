"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def construct():
    n = 256
    # Generate primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    prime_to_index = {p: i for i, p in enumerate(primes)}
    
    # Precompute prime masks for each number 1..n
    # Mask is an integer where bit i is set if prime[i] divides the number
    masks = [0] * (n + 1)
    masks[1] = 0
    for x in range(2, n + 1):
        temp = x
        m = 0
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                m |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is a prime factor
            if temp in prime_to_index:
                m |= (1 << prime_to_index[temp])
        masks[x] = m
        
    # Numbers sorted descending
    numbers = list(range(n, 0, -1))
    
    # Suffix sums for pruning
    suffix_sums = [0] * (len(numbers) + 1)
    for i in range(len(numbers) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + numbers[i]
        
    best_sum = 0
    best_set = []
    
    def solve(idx, current_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        # Pruning: even if we take all remaining, we can't beat best
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        if idx == len(numbers):
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set.copy()
            return
            
        x = numbers[idx]
        x_mask = masks[x]
        
        # Option 1: Include x if coprime to current selection
        if (current_mask & x_mask) == 0:
            current_set.append(x)
            solve(idx + 1, current_mask | x_mask, current_sum + x, current_set)
            current_set.pop()
            
        # Option 2: Exclude x
        solve(idx + 1, current_mask, current_sum, current_set)

    # Initial greedy lower bound to help pruning
    # Greedy: pick largest coprime numbers
    greedy_mask = 0
    greedy_sum = 0
    greedy_set = []
    for x in numbers:
        if (greedy_mask & masks[x]) == 0:
            greedy_mask |= masks[x]
            greedy_sum += x
            greedy_set.append(x)
    best_sum = greedy_sum
    best_set = greedy_set.copy()
    
    # Run Branch and Bound
    # To speed up, we can limit the search or rely on the strong pruning.
    # For N=256, this should be fast enough.
    solve(0, 0, 0, [])
    
    # Ensure 1 is included? 
    # 1 is coprime to everything. It adds 1 to the sum.
    # The B&B includes 1 as a candidate number.
    # Since 1 has mask 0, it will always be included in the optimal solution 
    # unless it prevents something else? No, it doesn't prevent anything.
    # So it will be in best_set.
    
    # Convert to list of ints
    result = best_set
    # Ensure types are Python int
    result = [int(x) for x in result]
    
    return result

# EVOLVE-BLOCK-END
