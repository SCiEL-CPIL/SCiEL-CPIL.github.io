"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, limit + 1) if is_prime[i]]

_PRIMES = _sieve_primes(500000)

def _factorize(num):
    factors = {}
    if num <= 1:
        return factors
    temp = num
    for p in _PRIMES:
        if p * p > temp:
            break
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp > 1:
        factors[temp] = factors.get(temp, 1) + 1
    return factors

def _divisors_from_factors(factors):
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _find_solution(n, max_x_trials=200):
    """Find distinct x,y,z > 0 with 1/x+1/y+1/z = 4/n, all <= 10^12."""
    x_min = n // 4 + 1
    x_max = (3 * n) // 4 + 1
    
    for x in range(x_min, min(x_max + 1, x_min + max_x_trials)):
        a = 4 * x - n
        if a <= 0:
            continue
        b = n * x
        
        # (a*y - b)(a*z - b) = b^2
        # Let u = a*y - b, v = a*z - b, then u*v = b^2
        # y = (u + b)/a, z = (v + b)/a
        # Need u, v > 0, u != v, y, z integers, y != z, y != x, z != x
        
        b_sq = b * b
        fac = _factorize(b_sq)
        divs = _divisors_from_factors(fac)
        
        for u in divs:
            if u >= b_sq:  # avoid duplicates and ensure u < v
                break
            v = b_sq // u
            if u * v != b_sq:
                continue
            if u == v:
                continue
            
            # y = (u + b) / a, z = (v + b) / a
            num_y = u + b
            num_z = v + b
            
            if num_y % a != 0 or num_z % a != 0:
                continue
            
            y = num_y // a
            z = num_z // a
            
            if y <= 0 or z <= 0:
                continue
            if y == z:
                continue
            if x == y or x == z:
                continue
            if y > 10**12 or z > 10**12:
                continue
            
            # Verify via cross multiplication
            # 1/x + 1/y + 1/z = 4/n
            # n*(yz + xz + xy) = 4*x*y*z
            lhs = n * (y * z + x * z + x * y)
            rhs = 4 * x * y * z
            if lhs != rhs:
                continue
            
            return [x, y, z]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = _find_solution(n, max_x_trials=30)
        
        if found is None:
            remaining = deadline - time.time()
            if remaining > 1.0:
                found = _find_solution(n, max_x_trials=150)
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
