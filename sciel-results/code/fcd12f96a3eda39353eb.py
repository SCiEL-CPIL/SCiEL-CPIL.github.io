import math
import time
import random

DEADLINE = time.time() + 110.0  # safety margin before 120s limit

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic Miller-Rabin for n < 3,474,749,660,383
    for a in [2, 3, 5, 7, 11, 13, 17]:
        if a >= n:
            continue
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randint(2, n - 2)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def largest_prime_factor(n, p_limit):
    """Return largest prime factor of n, or None if all factors <= p_limit."""
    if n <= 1:
        return 1
    # Try small primes up to 1000
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    for p in small_primes:
        if p > p_limit:
            break
        while n % p == 0:
            n //= p
    if n == 1:
        return 1
    if n <= p_limit:
        return n
    # Use Pollard Rho to factor remaining part
    factors = []
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            factors.append(m)
        else:
            d = pollard_rho(m)
            stack.append(d)
            stack.append(m // d)
    max_f = max(factors)
    return max_f

def construct():
    best_k = 0
    best_p = 2
    
    # Search primes up to some limit
    # We'll check primes in increasing order, but also try larger primes
    # The key insight: we need p^2+1, p^2+2, ..., p^2+k to have all prime factors <= p
    
    # First, let's try small primes efficiently
    for p in range(3, 10000, 2):
        if time.time() > DEADLINE:
            break
        if not is_prime(p):
            continue
        k = 0
        while k < 200:
            if time.time() > DEADLINE:
                break
            n = p * p + k + 1
            lp = largest_prime_factor(n, p)
            if lp is not None and lp > p:
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
    
    # Try larger primes if time permits
    # For larger p, p^2 is huge, so we need to be careful
    # But the condition is that all prime factors of p^2+i are <= p
    # This means p^2+i must be p-smooth
    
    # Try some specific larger primes
    test_primes = [10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093, 
                   2000003, 2000009, 2000017, 2000023, 2000029, 2000033, 2000039, 2000047, 2000051, 2000057,
                   1000000007]  # large prime
    
    for p in test_primes:
        if time.time() > DEADLINE:
            break
        if p > 10**9:
            continue
        if not is_prime(p):
            continue
        k = 0
        while k < 200:
            if time.time() > DEADLINE:
                break
            n = p * p + k + 1
            lp = largest_prime_factor(n, p)
            if lp is not None and lp > p:
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we haven't found anything better, return the best we have
    if best_k < 1:
        best_k = 1
        best_p = 2
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
