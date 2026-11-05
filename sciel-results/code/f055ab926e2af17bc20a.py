"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    answers = [None] * 254  # Index 0 corresponds to n=3, index 253 to n=256
    
    # Precompute Smallest Prime Factor (SPF) up to max B
    # Max n = 256. Max x < 3n/4 < 192.
    # Max B = n * x < 256 * 192 = 49152.
    # Let's sieve up to 50000 to be safe.
    limit = 50000
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_divisors_of_b_squared(n, x):
        B = n * x
        if B > limit:
            # Fallback to trial division if B exceeds sieve limit (unlikely)
            # But for n<=256, x<200, B is always < 51200
            pass
            
        # Factorize B
        factors = {}
        temp = B
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors[p] = count
        
        # Generate divisors of B^2
        # If B = prod p^e, B^2 = prod p^(2e)
        divisors = [1]
        for p, e in factors.items():
            exp = 2 * e
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        
        return divisors

    for idx in range(254):
        n = idx + 3
        
        # Check time
        if time.time() > deadline:
            break
            
        found = None
        
        # x < y < z
        # 1/x + 1/y + 1/z = 4/n
        # 1/x < 4/n => x > n/4
        # 3/x > 4/n => x < 3n/4
        
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # Get divisors of B^2
            divs = get_divisors_of_b_squared(n, x)
            
            # We need u * v = B^2 with u <= v (so u <= B)
            # And (u+B) % A == 0 and (v+B) % A == 0
            # And x < y < z
            
            for u in divs:
                if u > B:
                    continue
                v = (B * B) // u
                
                # Check divisibility
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                # Check distinctness and order
                if x < y < z:
                    # Check bound
                    if z <= 1000000000000:
                        found = [x, y, z]
                        break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
# EVOLVE-BLOCK-END
