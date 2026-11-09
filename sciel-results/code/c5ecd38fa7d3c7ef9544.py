"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin
    
    N = 256
    
    # Build list of all numbers 1..256 with their prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # Map each number to its set of prime factors
    pf = {1: set()}
    for i in range(2, N + 1):
        pf[i] = prime_factors(i)
    
    # All primes up to 256
    primes = [i for i in range(2, N + 1) if pf[i] == {i}]
    
    # Better initial solution: for each prime, pick the largest power of that prime <= 256
    # This uses only that one prime factor, so no conflicts between different primes' choices
    initial = {1}
    for p in primes:
        # Find largest power of p <= 256
        best_power = p
        val = p
        while val * p <= N:
            val *= p
            best_power = val
        initial.add(best_power)
    
    best = set(initial)
    best_sum = sum(best)
    
    # Track which primes are "used" (appear in any selected number's factorization)
    def used_primes(selected_set):
        up = set()
        for x in selected_set:
            up |= pf[x]
        return up
    
    # Fast validity check: a number can be added if its prime factors don't intersect used primes
    def can_add(num, used_set):
        if num == 1:
            return True
        return len(pf[num] & used_set) == 0
    
    # Local search: try adding a number, or swapping one number for another
    all_numbers = list(range(1, N + 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        up = used_primes(best)
        
        # Try adding each unselected number
        for num in all_numbers:
            if num in best:
                continue
            if can_add(num, up):
                best.add(num)
                best_sum += num
                up |= pf[num]
                improved = True
        
        # Try swapping: remove one, add another
        for remove_num in list(best):
            if remove_num == 1:
                continue  # Keep 1 always
            base_sum = best_sum - remove_num
            # Compute used primes without remove_num
            up_without = used_primes(best - {remove_num})
            for add_num in all_numbers:
                if add_num in best:
                    continue
                if add_num == remove_num:
                    continue
                if can_add(add_num, up_without):
                    new_sum = base_sum + add_num
                    if new_sum > best_sum:
                        best.discard(remove_num)
                        best.add(add_num)
                        best_sum = new_sum
                        up = used_primes(best)
                        improved = True
    
    # Convert to list of Python ints
    result = list(best)
    result.sort()
    # Ensure all are Python int
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
