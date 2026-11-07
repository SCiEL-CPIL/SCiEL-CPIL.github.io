"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Generate primes up to 40
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(2, 41) if is_prime(p)]
    
    candidates = []
    for p in primes:
        power = p
        while power <= 40:
            candidates.append(power)
            power *= p
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # We have candidates which are prime powers.
    # They are all distinct and multiplicatively independent.
    # We need at most 18.
    if len(candidates) > 18:
        result = candidates[:18]
    else:
        result = candidates
        
    # Ensure they are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
