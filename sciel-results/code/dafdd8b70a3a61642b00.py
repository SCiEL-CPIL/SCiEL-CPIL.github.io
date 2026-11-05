"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 55.0  # seconds, safety margin before 60s limit

def _check_violation_bitset(candidate, a_idx):
    """
    Check if element at a_idx is the mean of any subset of at least 2 OTHER elements.
    Uses bitset DP for speed.
    """
    a = candidate[a_idx]
    others = candidate[:a_idx] + candidate[a_idx+1:]
    n = len(others)
    if n < 2:
        return False
    
    # Transform: z_i = y_i - a
    zs = [y - a for y in others]
    
    # Calculate max possible sum magnitude for offset
    # Sum of absolute values is a safe upper bound for the range of subset sums
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        # All others are equal to a, but they are distinct, so this shouldn't happen unless n=0
        return False
        
    offset = max_sum
    # Bitset size: 2 * max_sum + 1
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitmask where bit s is set if sum s-offset is achievable with k elements
    # We need to track up to n elements
    dp = [0] * (n + 1)
    dp[0] = 1 << offset  # Sum 0 is achievable with 0 elements
    
    for z in zs:
        # Update backwards to avoid using the same element twice
        # We only need to update up to the number of elements processed so far
        # But simple loop is fine for n <= 48
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            dp[k] |= (shifted & mask)
            
            # Early exit: if we can form sum 0 (bit offset) with k >= 2
            # We can check this at the end, or during. Checking during might save time.
            # But we must ensure k >= 2.
            if k >= 2 and (dp[k] & (1 << offset)):
                return True

    # Final check for any k >= 2
    for k in range(2, n + 1):
        if dp[k] & (1 << offset):
            return True
            
    return False

def _full_check(candidate):
    """Check all elements for violations."""
    for i in range(len(candidate)):
        if _check_violation_bitset(candidate, i):
            return True
    return False

def construct():
    start_time = time.time()
    
    best_set = []
    
    # Define multiple orders to try
    # 1. Increasing
    # 2. Decreasing
    # 3. Center-out (closest to 256 first)
    # 4. Odds then Evens
    # 5. Random permutations (seeded)
    
    orders = []
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    # Center out: sort by distance from 256.5
    orders.append(sorted(range(1, 513), key=lambda x: abs(x - 256.5)))
    
    # Odds then evens
    orders.append(list(range(1, 513, 2)) + list(range(2, 513, 2)))
    
    # Evens then odds
    orders.append(list(range(2, 513, 2)) + list(range(1, 513, 2)))
    
    # Primes first
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
    primes = [x for x in range(1, 513) if is_prime(x)]
    composites = [x for x in range(1, 513) if x not in primes]
    orders.append(primes + composites)
    orders.append(composites + primes)

    # Add some random orders
    rng = random.Random(42)
    for _ in range(10):
        rand_order = list(range(1, 513))
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    # Function to run greedy for a given order
    def run_greedy(order):
        selected = []
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                return selected
            
            candidate = selected + [x]
            new_idx = len(selected)
            
            # Check if new element x is the mean of any subset of selected
            # If selected has < 2 elements, x cannot be the mean of a subset of size >= 2
            if len(selected) >= 2:
                if _check_violation_bitset(candidate, new_idx):
                    continue
            
            # Check if any existing element is now the mean of a subset involving x
            # We can optimize this:
            # 1. Check if x is mean of subset of selected (done above)
            # 2. Check if any y in selected is mean of subset of (selected + x)
            #    This is equivalent to checking if y is mean of subset of selected that includes x?
            #    No, the subset can be ANY subset of the new set that does NOT include y.
            #    If the subset does not include x, it was already checked.
            #    So we only need to check subsets that INCLUDE x.
            #    Let S be the subset. S contains x. y is the mean of S.
            #    y = (x + sum(S\{x})) / (1 + |S\{x}|)
            #    (1 + k)y = x + sum(S\{x}) where k = |S\{x}| >= 1.
            #    sum(S\{x}) = (1+k)y - x.
            #    We need to check if there is a subset T of selected (excluding y) of size k (k>=1)
            #    such that sum(T) = (1+k)y - x.
            
            # This is a specific subset sum problem for each y and each k.
            # Given the small size of selected (max 48), we can just do a full check for now
            # but we can skip the check if len(selected) == 0 or 1.
            
            if _full_check(candidate):
                continue
            
            selected.append(x)
            
            if len(selected) >= 48:
                break
                
        return selected

    # Run predefined orders
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        current_set = run_greedy(order)
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if len(best_set) >= 48:
            break
            
    # If we have time, run more random restarts
    # This helps escape local optima of the greedy approach
    rng2 = random.Random(123)
    while time.time() - start_time < _DEADLINE - 10:
        # Generate a new random order
        # We can bias the random order slightly towards center or specific patterns
        rand_order = list(range(1, 513))
        rng2.shuffle(rand_order)
        
        current_set = run_greedy(rand_order)
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if len(best_set) >= 48:
            break

    # Final verification of best_set
    if best_set:
        if _full_check(best_set):
            # Fallback: return a known valid set
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
