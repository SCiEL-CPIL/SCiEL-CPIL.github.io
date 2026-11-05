import time
import math
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N_MAX = 1000000
    MIN_N = 10000
    
    # Precompute primes for factorization of n and n+1
    # We need primes up to sqrt(1000000) = 1000
    limit_primes = 1000
    is_prime = np.ones(limit_primes + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit_primes)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    def get_powerful_part(x):
        if x <= 1:
            return 1
        p_part = 1
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count >= 2:
                    p_part *= (p ** count)
        if temp > 1:
            # temp is prime, exponent 1
            pass
        return p_part

    best_score = 0
    best_n = MIN_N
    
    # Search for A, B
    # A^2 divides n, B^2 divides n+1
    # gcd(A, B) must be 1
    # Maximize A^2 B^2
    
    # We can limit A, B to range where A^2 B^2 is potentially > best_score
    # Initially best_score is 0, so we check all.
    # To speed up, we can iterate A, B in descending order of A^2 B^2?
    # Or just nested loops.
    
    # Let's try A from 1000 down to 10
    # B from 1000 down to 10
    # This is 100k iterations.
    
    for A in range(1000, 9, -1):
        A2 = A * A
        # If A2 * 100 * 100 < best_score, we can break?
        # Max B is 1000, so max B2 is 10^6.
        # If A2 * 10^6 < best_score, then for this A and any B, score < best.
        # But B is limited by gcd.
        # Rough check:
        if A2 * 1000000 < best_score:
            break
            
        for B in range(1000, 9, -1):
            B2 = B * B
            potential = A2 * B2
            if potential < best_score:
                continue # Since B is descending, smaller B will have smaller potential
            
            if gcd(A, B) != 1:
                continue
                
            # Solve n = A2 * x
            # A2 * x = -1 (mod B2)
            # x = -A2^(-1) (mod B2)
            
            # Find inverse of A2 mod B2
            # Since gcd(A, B) = 1, gcd(A2, B2) = 1
            try:
                inv = pow(A2, -1, B2)
            except ValueError:
                continue # Should not happen if gcd is 1
                
            x0 = (-inv) % B2
            
            # Smallest positive n
            n0 = A2 * x0
            
            # Check if n0 or n0 + k * M is in [MIN_N, N_MAX]
            M = A2 * B2
            
            # If n0 > N_MAX, no solution
            if n0 > N_MAX:
                continue
                
            # If n0 < MIN_N, find smallest k such that n0 + k*M >= MIN_N
            if n0 < MIN_N:
                k = (MIN_N - n0 + M - 1) // M
                n_candidate = n0 + k * M
            else:
                n_candidate = n0
                
            if n_candidate > N_MAX:
                continue
                
            # Calculate actual score
            p_n = get_powerful_part(n_candidate)
            p_np1 = get_powerful_part(n_candidate + 1)
            score = p_n * p_np1
            
            if score > best_score:
                best_score = score
                best_n = n_candidate
                
            if time.time() > deadline - 5:
                return int(best_n)
                
    return int(best_n)
