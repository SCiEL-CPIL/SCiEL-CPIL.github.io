"""Precompute all solutions for 1/x + 1/y + 1/z = 4/n for n=3..256."""
import math
import time

def _sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

# n <= 256, x can be up to ~4n/3 < 342, so B = n*x < 256*342 ~ 87552
# B^2 can be up to ~7.6 * 10^9, need primes up to sqrt of that ~ 87000
PRIMES = _sieve_primes(100000)

def _factorize(num):
    if num <= 1:
        return []
    factors = []
    temp = num
    for p in PRIMES:
        if p * p > temp:
            break
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _divisors_from_factorization(prime_exp_pairs):
    divisors = [1]
    for p, e in prime_exp_pairs:
        current_len = len(divisors)
        pow_val = 1
        for _ in range(e):
            pow_val *= p
            for i in range(current_len):
                divisors.append(divisors[i] * pow_val)
    return divisors

def _find_solution(n):
    """Find x, y, z distinct positive integers <= 10^12 with 1/x+1/y+1/z = 4/n."""
    BOUND = 10**12
    
    # Try x values: start with x = n, then nearby values
    # x must satisfy: 4x - n > 0 => x > n/4
    # and for y, z to be positive and distinct, we need reasonable bounds
    # From 1/x + 1/y + 1/z = 4/n, if x is the smallest, then 3/x >= 4/n => x <= 3n/4
    # If x is the largest, 1/x <= 4/n - 2/x_min... let's just try a range
    
    # Strategy: try x in a smart order
    # First try x = n, then x = n±1, n±2, etc., also try small x
    x_candidates = []
    
    # x must be > n/4 and for the equation to have solutions with y,z > x
    # 1/y + 1/z = 4/n - 1/x = (4x-n)/(nx)
    # Need 4x - n > 0, so x > n/4
    # Also need y, z > x for distinctness (assuming x < y < z)
    # 1/y + 1/z < 2/x, so (4x-n)/(nx) < 2/x => 4x-n < 2n => 4x < 3n => x < 3n/4
    
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    if x_max < x_min:
        x_max = x_min + 10  # fallback
    
    # Try x = n first, then nearby
    candidates = [n]
    for d in range(1, 50):
        if n - d >= x_min:
            candidates.append(n - d)
        if n + d <= x_max:
            candidates.append(n + d)
    
    # Also add some other candidates
    for x in range(x_min, min(x_min + 20, x_max + 1)):
        if x not in candidates:
            candidates.append(x)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen and x_min <= c <= x_max:
            seen.add(c)
            unique_candidates.append(c)
    
    for x in unique_candidates:
        A = 4 * x - n
        B = n * x
        if A <= 0:
            continue
        
        # Need u*v = B^2 where u = A*y - B, v = A*z - B
        # y = (u + B) / A, z = (v + B) / A
        # Need u + B ≡ 0 (mod A) and v + B ≡ 0 (mod A)
        # u * v = B^2
        # So u ≡ -B (mod A) and B^2/u ≡ -B (mod A)
        
        # Factorize B
        factors_B = _factorize(B)
        factors_B2 = [(p, 2 * e) for p, e in factors_B]
        
        # Get divisors of B^2
        divs = _divisors_from_factorization(factors_B2)
        
        for u in divs:
            if u >= B:
                continue
            if (u + B) % A != 0:
                continue
            y = (u + B) // A
            if y <= x:
                continue
            
            v = (B * B) // u
            if (v + B) % A != 0:
                continue
            z = (v + B) // A
            
            if z <= y:
                continue
            
            if z > BOUND:
                continue
            
            # Verify
            if x * y * z * 4 == n * (y * z + x * z + x * y):
                return [x, y, z]
    
    return None

# Precompute all solutions at module load time
_SOLUTIONS = {}
_start = time.time()
for _n in range(3, 257):
    _SOLUTIONS[_n] = _find_solution(_n)
_elapsed = time.time() - _start

def construct():
    """Return 254 rows for n=3..256. Each row is None or [x, y, z]."""
    return [_SOLUTIONS[n] for n in range(3, 257)]
