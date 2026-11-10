"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline for 90s wall-clock limit
    
    answers = []
    
    # Precompute primes for factorization up to sqrt(max B)
    # Max n=256, max x=192. Max B = 256*192 = 49152.
    # sqrt(49152) ~ 221.
    # Let's generate primes up to 300.
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(300)
    
    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

    def get_divisors(factors):
        # Returns list of divisors
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divisors = new_divs
        return divisors

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: x > n/4 and x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let U = Ay - B, V = Az - B
            # U * V = B^2
            # U must be a divisor of B^2
            # Also y = (U + B) / A must be integer => U + B divisible by A => U % A == (-B) % A
            # Similarly for z
            
            # Factor B to get divisors of B^2
            # B = n * x
            # We can just factor B and double the exponents for B^2
            factors_B = factorize(B)
            factors_B2 = {p: 2*e for p, e in factors_B.items()}
            
            divs = get_divisors(factors_B2)
            
            # Optimization: We only need U such that U <= V (since y <= z implies U <= V)
            # U * V = B^2 => U <= B
            # Also y > x => (U+B)/A > x => U+B > Ax => U > Ax - B = A(x) - B
            # Ax - B = x(4x-n) - nx = 4x^2 - nx - nx = 4x^2 - 2nx = 2x(2x - n)
            # So U > 2x(2x - n)
            
            min_U = 2 * x * (2 * x - n)
            if min_U < 1:
                min_U = 1
                
            target_mod = (-B) % A
            
            for U in divs:
                if U > B:
                    continue
                if U <= min_U:
                    continue
                
                if U % A != target_mod:
                    continue
                
                V = (B * B) // U
                if V % A != target_mod:
                    continue
                
                y = (U + B) // A
                z = (V + B) // A
                
                # Check distinctness and ordering
                # We enforced y > x via min_U
                # We enforced y <= z via U <= B (which implies U <= V)
                # If U == V, y == z, which is not distinct.
                if y == z:
                    continue
                if x == y or x == z:
                    continue
                if z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
