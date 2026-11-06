import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50
    TARGET = 123457
    
    # Generate primes up to TARGET (we might need primes up to TARGET for checking)
    # Actually, we only need primes up to TARGET for the smooth number generation
    # But for the binary search, we only need to try primes up to ~23 initially
    
    # First, let's try small primes in order
    # Primes to try: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, ...
    
    def generate_smooth_numbers(B, limit):
        """Generate all B-smooth numbers up to limit.
        B-smooth means all prime factors are <= B."""
        # Get primes up to B
        if B < 2:
            return [1]
        
        # Sieve for primes up to B
        is_prime = [True] * (B + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(B)) + 1):
            if is_prime[i]:
                for j in range(i*i, B + 1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, B + 1) if is_prime[i]]
        
        # Generate all numbers of form p1^a1 * p2^a2 * ... * pk^ak <= limit
        smooth = set()
        
        def gen(idx, current):
            if current > limit:
                return
            if idx == len(primes):
                smooth.add(current)
                return
            p = primes[idx]
            val = current
            while val <= limit:
                gen(idx + 1, val)
                if val > limit // p:
                    break
                val *= p
        
        gen(0, 1)
        return smooth
    
    # Try primes in increasing order
    # Start with small primes
    small_primes_to_try = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_a = 1
    best_b = TARGET - 1
    best_max = TARGET  # P(123456) = 23, so this is an upper bound
    
    for B in small_primes_to_try:
        if time.time() > deadline:
            break
        
        smooth_set = generate_smooth_numbers(B, TARGET - 1)
        
        # Check if any a in smooth_set has (TARGET - a) also in smooth_set
        found = False
        for a in smooth_set:
            if a < 1 or a >= TARGET:
                continue
            b = TARGET - a
            if b in smooth_set:
                # Both a and b are B-smooth
                # The actual max prime factor could be less than B, but B is an upper bound
                # We want to verify the actual P(a) and P(b)
                # But since we're going in increasing B order, the first B that works gives us
                # max(P(a), P(b)) <= B, and since no smaller B worked, the answer is exactly B
                # (or possibly a prime less than B but greater than the previous tried prime)
                best_a = a
                best_b = b
                best_max = B
                found = True
                break
        
        if found:
            break
    
    # If we found a pair with B-smoothness, verify and return
    if best_max < TARGET:
        return [best_a, best_b]
    
    # Fallback: use the known good pair
    return [1, TARGET - 1]
