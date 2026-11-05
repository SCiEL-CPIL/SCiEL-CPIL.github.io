import time
import math
import sys

def construct():
    """
    Find distinct three-term unit-fraction representations of 4/n for n=3..256.
    Returns a list of 254 rows, each either None or [x, y, z] with
    1/x + 1/y + 1/z = 4/n, x < y < z, all positive integers <= 10^12.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    answers = [None] * 254  # Index 0 -> n=3, index 253 -> n=256
    
    # Precompute primes up to a reasonable limit for factorization
    # We need to factor numbers up to ~n * x_max. For n=256, if x ~ 1000, B ~ 256000.
    # But x could be larger. Let's use a generous prime list.
    # Actually, we'll factor on the fly with trial division using primes up to sqrt(N).
    # Let's precompute primes up to 100000 for speed.
    
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]
    
    primes = sieve_primes(1000000)
    
    def factorize(n):
        """Return list of (prime, exponent) pairs for n."""
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        return factors
    
    def get_divisors_from_factorization(factors):
        """Generate all divisors from a factorization list."""
        divisors = [1]
        for p, exp in factors:
            current = []
            power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    current.append(d * power)
                power *= p
            divisors = current
        return divisors
    
    def get_divisors(n):
        """Get all divisors of n."""
        if n <= 0:
            return []
        factors = factorize(n)
        return get_divisors_from_factorization(factors)
    
    # For each n, find a valid triple (x, y, z) with x < y < z
    # 1/x + 1/y + 1/z = 4/n
    # Strategy: iterate over possible x values, then for each x,
    # solve for y and z using the divisor method.
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        # x must satisfy 1/x < 4/n, so x > n/4
        # Also x < y < z, so 3/x > 4/n, thus x < 3n/4
        # More precisely: since x < y < z, we have 1/x > 1/y > 1/z
        # So 1/x < 4/n (since 1/x is the largest term, but sum is 4/n, 
        # actually 1/x < 4/n is not quite right. Let's think:
        # 1/x + 1/y + 1/z = 4/n, and x < y < z means 1/x > 1/y > 1/z
        # So 3*(1/x) > 4/n => x < 3n/4
        # And 1/x < 4/n => x > n/4
        # So x is in range (n/4, 3n/4)
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in this range
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            # Let A = 4x - n, B = nx
            # We need A > 0, so x > n/4 (already ensured)
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # 1/y + 1/z = A/B
            # (y+z)/(yz) = A/B
            # B(y+z) = A*yz
            # A*yz - B*y - B*z = 0
            # (Ay - B)(Az - B) = B^2
            # Let u = Ay - B, v = Az - B, then u*v = B^2
            # y = (u + B)/A, z = (v + B)/A
            # We need y > x, z > y, and u, v positive divisors of B^2
            # Also u < v since y < z
            
            # Find divisors of B^2
            # B can be large. n up to 256, x up to ~192, so B up to ~49152
            # B^2 up to ~2.4e9, which is manageable for divisor enumeration
            # if we factorize B first
            
            # Factorize B
            B_factors = factorize(B)
            # B^2 has factors: for each (p, e) in B_factors, B^2 has (p, 2e)
            B2_factors = [(p, 2*e) for p, e in B_factors]
            B2_divisors = get_divisors_from_factorization(B2_factors)
            
            # For each divisor u of B^2 with u < B^2/u (i.e., u < B),
            # check if y = (u + B)/A is an integer > x
            # and z = (B^2/u + B)/A is an integer > y
            
            B2 = B * B
            
            for u in B2_divisors:
                if u * u > B2:
                    break
                if B2 % u != 0:
                    continue
                v = B2 // u
                if u >= v:
                    continue
                
                # y = (u + B) / A must be integer
                if (u + B) % A != 0:
                    continue
                y = (u + B) // A
                if y <= x:
                    continue
                
                # z = (v + B) / A must be integer
                if (v + B) % A != 0:
                    continue
                z = (v + B) // A
                if z <= y:
                    continue
                
                # Check bounds
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                
                # Verify the solution
                # 1/x + 1/y + 1/z = 4/n
                # Cross multiply: n*(y*z + x*z + x*y) = 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
