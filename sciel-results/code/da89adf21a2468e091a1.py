"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp
    
    def generate_smooth(prime_list, limit):
        """Generate all numbers <= limit with all prime factors in prime_list,
        returning dict {number: largest_prime_factor}"""
        smooth = {}
        def gen(idx, current, max_pf):
            if time.time() > deadline:
                return
            if current > limit:
                return
            if current > 1:
                smooth[current] = max_pf
            if idx >= len(prime_list):
                return
            p = prime_list[idx]
            # Try using this prime 0, 1, 2, ... times
            val = current
            for _ in range(20):  # 2^20 > 1e6, more than enough
                gen(idx + 1, val, max_pf)
                if val > limit // p:
                    break
                val *= p
        gen(0, 1, 1)
        return smooth
    
    # Try increasing prime bounds
    prime_bounds = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307]
    
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307]
    
    best_score = float('inf')
    best_a = 1
    best_b = TARGET - 1
    
    for max_p in prime_bounds:
        if time.time() > deadline - 5:
            break
        # Get primes up to max_p
        primes = [p for p in all_primes if p <= max_p]
        smooth = generate_smooth(primes, TARGET)
        
        for a, pa in smooth.items():
            if time.time() > deadline - 2:
                break
            b = TARGET - a
            if b < 1:
                continue
            pb = largest_prime_factor(b)
            score = max(pa, pb)
            if score < best_score:
                best_score = score
                best_a = a
                best_b = b
                if best_score == 1:
                    break
        if best_score == 1:
            break
        if best_score <= max_p:
            # If we found a solution where both have PF <= max_p, we can stop
            # since increasing max_p won't help
            break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
