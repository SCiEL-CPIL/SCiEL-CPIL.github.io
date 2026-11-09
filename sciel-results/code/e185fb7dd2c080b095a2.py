import time
import math
import random
import sys

def construct():
    """
    Find a large subset S of [1, 512] such that lcm(a, b) <= 512 for all a, b in S.
    Uses multi-start greedy construction with local search.
    """
    start_time = time.time()
    LIMIT = 512
    N = LIMIT
    deadline = start_time + 85  # Safety margin

    # Precompute compatibility
    # compat[i] is a bitmask of j in [1, 512] such that lcm(i, j) <= 512
    # We use 0-indexed bits for numbers 1..512. Bit i corresponds to number i+1.
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            # Optimization: if i*j > LIMIT * gcd(i,j), then lcm > LIMIT
            # lcm(i, j) = i * j // gcd(i, j)
            # i * j <= LIMIT * gcd(i, j)
            g = math.gcd(i, j)
            if (i * j) <= LIMIT * g:
                mask |= (1 << (j - 1))
        compat[i] = mask

    def mask_to_list(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]

    def list_to_mask(lst):
        mask = 0
        for x in lst:
            mask |= (1 << (x - 1))
        return mask

    def is_compatible_with_all(x, S_mask):
        # Check if x is compatible with all elements in S_mask
        # S_mask contains bits for elements in S
        # We need (compat[x] & S_mask) == S_mask
        return (compat[x] & S_mask) == S_mask

    def greedy_build(order):
        """
        Greedily build a set S by iterating through 'order' and adding
        elements if they are compatible with current S.
        """
        S_mask = 0
        for x in order:
            if is_compatible_with_all(x, S_mask):
                S_mask |= (1 << (x - 1))
        return S_mask

    def local_search(S_mask):
        """
        Try to improve S_mask by adding compatible elements or swapping.
        """
        improved = True
        while improved:
            improved = False
            # 1. Try to add any compatible element
            # Iterate through all x not in S_mask
            # To speed up, we can just check all 1..512
            for x in range(1, N + 1):
                if not (S_mask & (1 << (x - 1))):
                    if is_compatible_with_all(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        improved = True
                        # Restart loop to check for more additions
                        break
            
            if improved:
                continue

            # 2. Try swaps
            # For each y in S, try to replace with z not in S
            # such that z is compatible with S \ {y}
            S_list = mask_to_list(S_mask)
            for y in S_list:
                S_without_y = S_mask & ~(1 << (y - 1))
                # Find a z not in S that is compatible with S_without_y
                # We prefer z with high "potential" (e.g., high degree in compat graph)
                # For now, just find any such z
                # Optimization: Precompute a list of candidates?
                # Just iterate
                found_z = False
                for z in range(1, N + 1):
                    if not (S_mask & (1 << (z - 1))):
                        if is_compatible_with_all(z, S_without_y):
                            # Swap y with z
                            S_mask = S_without_y | (1 << (z - 1))
                            improved = True
                            found_z = True
                            break
                if found_z:
                    break
        
        return S_mask

    # Define different ordering strategies
    all_numbers = list(range(1, N + 1))
    
    # Strategy 1: Ascending value
    order1 = all_numbers
    
    # Strategy 2: Descending number of divisors
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    order2 = sorted(all_numbers, key=lambda x: num_divisors(x), reverse=True)
    
    # Strategy 3: Ascending max prime factor
    def max_prime_factor(n):
        if n <= 1:
            return 0
        mfp = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                mfp = d
                n //= d
            d += 1
        if n > 1:
            mfp = n
        return mfp

    order3 = sorted(all_numbers, key=lambda x: max_prime_factor(x))
    
    # Strategy 4: Random
    # We will generate random orders in the loop

    best_mask = 0
    best_size = 0
    
    # Run fixed strategies
    for order in [order1, order2, order3]:
        if time.time() > deadline:
            break
        S_mask = greedy_build(order)
        S_mask = local_search(S_mask)
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask

    # Run random strategies
    # Time budget for random restarts
    # Each restart takes some time. Let's estimate.
    # Greedy is O(N^2) approx. Local search is O(N^3) worst case but usually fast.
    # Let's run as many as possible.
    
    while time.time() < deadline - 5: # Leave 5s for final processing
        random_order = all_numbers[:]
        random.shuffle(random_order)
        S_mask = greedy_build(random_order)
        S_mask = local_search(S_mask)
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask

    result = mask_to_list(best_mask)
    
    # Fallback
    if not result:
        result = [1]
        
    return result

# Run construct to test
# print(construct())
