"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Generate all powerful numbers up to MAX_N
    # A powerful number is a product of prime powers p^e where e >= 2
    # We'll generate them by enumerating combinations of prime powers
    
    # First, generate all prime powers p^e <= MAX_N where e >= 2
    prime_powers = []  # list of (p, e, p^e)
    limit = int(math.isqrt(MAX_N))
    
    # Sieve to get primes up to sqrt(MAX_N)
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    for p in primes:
        pe = p * p
        e = 2
        while pe <= MAX_N:
            prime_powers.append((p, e, pe))
            pe *= p
            e += 1
    
    # Also include primes p where p^2 > MAX_N but p <= MAX_N (these can only appear as p^1, not powerful)
    # Wait, no. For a number to have p in its powerful part, we need p^e with e>=2 dividing it.
    # So we only need prime powers p^e with e>=2 and p^e <= MAX_N.
    
    # Now generate all powerful numbers as products of these prime powers with distinct primes
    powerful_numbers = {}  # value -> set of primes used
    
    def gen_powerful(idx, current_val, current_primes):
        if current_val > MAX_N:
            return
        if current_val >= MIN_N // 2:  # Early termination hint
            if current_val > 1:
                if current_val not in powerful_numbers:
                    powerful_numbers[current_val] = set(current_primes)
                else:
                    powerful_numbers[current_val].update(current_primes)
        
        for i in range(idx, len(prime_powers)):
            p, e, pe = prime_powers[i]
            if p in current_primes:
                continue
            new_val = current_val * pe
            if new_val > MAX_N:
                break
            gen_powerful(i + 1, new_val, current_primes | {p})
    
    gen_powerful(0, 1, set())
    
    # Sort powerful numbers by value descending to prioritize large A, B
    sorted_powerful = sorted(powerful_numbers.keys(), reverse=True)
    
    # For each pair (A, B) with disjoint prime supports, try to find S1, S2
    best_n = MIN_N
    best_val = 0
    
    # Limit the number of pairs we check to stay within time
    max_pairs_to_check = 5000000
    
    pairs_checked = 0
    for i, A in enumerate(sorted_powerful):
        if time.time() > deadline - 5:
            break
        primes_A = powerful_numbers[A]
        for j in range(i + 1, len(sorted_powerful)):
            B = sorted_powerful[j]
            if B < A:
                # Since sorted descending, once B < A, all further B are smaller
                # But we want A*B to be large, so let's continue but break if B is too small
                if A * B <= best_val:
                    break
            
            primes_B = powerful_numbers[B]
            if primes_A & primes_B:
                continue
            
            # We need B*S2 - A*S1 = 1, i.e., B*S2 ≡ 1 (mod A)
            # So S2 ≡ B^{-1} (mod A)
            try:
                inv_B_mod_A = pow(B, -1, A)
            except ValueError:
                # gcd(B, A) != 1, but we already checked disjoint primes
                continue
            
            # S2 = inv_B_mod_A + k*A for k = 0, 1, 2, ...
            # S1 = (B*S2 - 1) / A
            # We need S1 >= 1, S2 >= 1, both squarefree, and n = A*S1 in [MIN_N, MAX_N]
            
            # S2 must be positive. inv_B_mod_A is in [1, A-1] if A > 1
            if A == 1:
                continue
            
            S2_min = inv_B_mod_A
            if S2_min < 1:
                S2_min = 1
            
            # n = A*S1 = B*S2 - 1
            # We need MIN_N <= A*S1 <= MAX_N, so MIN_N <= B*S2 - 1 <= MAX_N
            # So (MIN_N + 1) / B <= S2 <= (MAX_N + 1) / B
            
            S2_lower = max(S2_min, (MIN_N + 1 + B - 1) // B)  # ceil((MIN_N+1)/B)
            S2_upper = (MAX_N + 1) // B
            
            if S2_lower > S2_upper:
                continue
            
            # Try a few values of S2
            for S2 in range(S2_lower, min(S2_upper + 1, S2_lower + 100)):
                if (B * S2 - 1) % A != 0:
                    continue
                S1 = (B * S2 - 1) // A
                if S1 < 1:
                    continue
                
                n = A * S1
                if n < MIN_N or n > MAX_N:
                    continue
                if n + 1 > MAX_N:
                    continue
                
                # Check S1 is squarefree and gcd(S1, A) = 1
                # Check S2 is squarefree and gcd(S2, B) = 1
                if gcd(S1, A) != 1 or gcd(S2, B) != 1:
                    continue
                
                # Check squarefree
                if not is_squarefree(S1) or not is_squarefree(S2):
                    continue
                
                val = A * B
                if val > best_val:
                    best_val = val
                    best_n = n
            
            pairs_checked += 1
            if pairs_checked >= max_pairs_to_check:
                break
    
    return best_n

def is_squarefree(n):
    """Check if n is squarefree."""
    if n <= 1:
        return True
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % (p * p) == 0:
            return False
    return True
# EVOLVE-BLOCK-END
