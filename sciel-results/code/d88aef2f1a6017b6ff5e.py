import math

def construct():
    # 1. Identify primes up to 256
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(2, 257) if is_prime(p)]
    
    # 2. Split primes into small (<= 16) and large (> 16)
    small_primes = [p for p in primes if p <= 16]
    large_primes = [p for p in primes if p > 16]
    
    # 3. Base sum includes 1 and all large primes
    base_sum = 1 + sum(large_primes)
    
    # 4. Backtracking to find best assignment for small primes
    # small_primes is [2, 3, 5, 7, 11, 13]
    # We want to assign x_p for each p in small_primes
    # x_p is a multiple of p, 1 <= x_p <= 256
    # gcd(x_p, x_q) == 1 for all p != q
    
    best_sum = 0
    # Order of processing: doesn't matter much, but let's use the list as is
    # To prune earlier, we might want to process primes with fewer options or larger impact first?
    # Actually, larger multiples conflict more.
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def backtrack(idx, chosen, current_sum):
        nonlocal best_sum
        if idx == len(small_primes):
            if current_sum > best_sum:
                best_sum = current_sum
            return
        
        p = small_primes[idx]
        # Try multiples of p from largest to smallest
        # Largest multiple is (256 // p) * p
        max_mult = (256 // p) * p
        for m in range(max_mult, 0, -p):
            # Check if m is coprime to all chosen
            valid = True
            for c in chosen:
                if gcd(m, c) != 1:
                    valid = False
                    break
            if valid:
                backtrack(idx + 1, chosen + [m], current_sum + m)
                
    backtrack(0, [], 0)
    
    # 5. Construct the final list
    # We need to retrieve the actual numbers, not just the sum.
    # Let's modify backtrack to store the best choice.
    
    best_choice = []
    best_sum_val = 0
    
    def backtrack2(idx, chosen, current_sum):
        nonlocal best_sum_val, best_choice
        if idx == len(small_primes):
            if current_sum > best_sum_val:
                best_sum_val = current_sum
                best_choice = chosen[:]
            return
        
        p = small_primes[idx]
        max_mult = (256 // p) * p
        for m in range(max_mult, 0, -p):
            valid = True
            for c in chosen:
                if gcd(m, c) != 1:
                    valid = False
                    break
            if valid:
                backtrack2(idx + 1, chosen + [m], current_sum + m)
                
    backtrack2(0, [], 0)
    
    result = [1] + large_primes + best_choice
    return result

