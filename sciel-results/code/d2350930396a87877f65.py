import time
from fractions import Fraction
import sys

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin for 120s limit
    
    primes = _primes_up_to(1000)
    # Sort primes, maybe smaller primes are better for density?
    # But budget is 1/p. Smaller p costs more.
    # Large p costs less but covers sparser.
    # We want to cover 1..L.
    
    budget = Fraction(1, 1)
    used_budget = Fraction(0, 1)
    used_primes = set()
    classes = []
    
    MAX_L = 100000
    # We need to track coverage up to MAX_L
    # Using a list of booleans or bytearray
    is_covered = bytearray(MAX_L + 10)
    
    # Precompute primes for quick access
    prime_list = primes
    
    current_L = 0
    
    # We will iterate while we have budget and time
    while time.time() < DEADLINE:
        # Find the first gap
        # current_L is the last covered integer
        # We need to check current_L + 1
        n = current_L + 1
        
        # If n > MAX_L, we are done
        if n > MAX_L:
            break
            
        # If n is already covered (should not happen if we maintain current_L correctly, 
        # but just in case of logic errors or if we jump)
        while n <= MAX_L and is_covered[n]:
            current_L = n
            n += 1
            
        if n > MAX_L:
            break
            
        # n is a gap. We need to cover it.
        # Find the best prime to add.
        best_p = None
        best_r = None
        best_ext = -1
        
        # Iterate through unused primes
        # Optimization: Only consider primes that fit in budget
        # Also, iterating all 168 primes every step might be slow if steps are many.
        # But steps are bounded by number of primes (168).
        # So 168 * 168 * (avg extension check) operations.
        # Avg extension check is small (until we hit a gap).
        
        for p in prime_list:
            if p in used_primes:
                continue
                
            cost = Fraction(1, p)
            if used_budget + cost > budget:
                continue
                
            r = n % p
            
            # Check extension
            # We want to know the first m >= n such that:
            # 1. Not covered by existing classes
            # 2. Not covered by new class (p, r)
            
            # We can break early if we find a gap
            # Limit the search to a reasonable window to avoid O(MAX_L) per candidate
            # If we don't find a gap in the first 1000 numbers, assume it's good?
            # No, we need to know the actual extension to compare.
            # But if all candidates extend by > 1000, we can just pick the first one?
            # Or pick the one with smallest p (densest)?
            
            # Let's do a bounded scan.
            # If scan hits limit, assign a high score.
            scan_limit = n + 500 
            if scan_limit > MAX_L:
                scan_limit = MAX_L
                
            m = n
            found_gap = False
            while m < scan_limit:
                if not is_covered[m] and (m % p != r):
                    # Gap found at m
                    ext = m - n
                    if ext > best_ext:
                        best_ext = ext
                        best_p = p
                        best_r = r
                    found_gap = True
                    break
                m += 1
                
            if not found_gap:
                # Covered up to scan_limit
                ext = scan_limit - n
                if ext > best_ext:
                    best_ext = ext
                    best_p = p
                    best_r = r
                    
        if best_p is None:
            # No prime fits budget or no improvement
            break
            
        # Commit
        used_budget += Fraction(1, best_p)
        used_primes.add(best_p)
        classes.append([best_p, best_r])
        
        # Update is_covered
        # Mark best_r, best_r + best_p, ...
        start_val = best_r
        if start_val == 0:
            start_val = best_p
            
        for val in range(start_val, MAX_L + 1, best_p):
            is_covered[val] = 1
            
        # Update current_L
        # We know n was covered. Let's find the new current_L
        # We can just let the while loop at the top handle it next iteration,
        # but it's faster to advance now.
        new_L = n
        while new_L <= MAX_L and is_covered[new_L]:
            new_L += 1
        current_L = new_L - 1
        
        if current_L >= MAX_L:
            break

    # Fallback if no classes
    if not classes:
        if primes:
            classes.append([primes[0], 0])
            
    return classes

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Classes: {len(res)}")
    # Calculate L
    L_max = 100000
    is_covered = bytearray(L_max + 1)
    for p, r in res:
        start = r
        if start == 0:
            start = p
        for i in range(start, L_max + 1, p):
            is_covered[i] = 1
            
    L = 0
    while L < L_max and is_covered[L]:
        L += 1
    print(f"L: {L-1}")
