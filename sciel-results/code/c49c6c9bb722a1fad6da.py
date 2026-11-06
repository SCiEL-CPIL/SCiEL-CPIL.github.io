import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    deadline = time.time() + 50  # Safety margin before 60s limit (or 90s kill)
    
    # We need the smallest n such that phi(n) % 120 == 0.
    # phi(n) = n * product_{p|n} (1 - 1/p)
    # 120 = 2^3 * 3 * 5
    
    # Let's just iterate n from 1 upwards and compute phi(n).
    # Since the answer is likely small, this is very fast.
    # However, to be robust, we can use a sieve if the loop takes too long,
    # but for n < 10000, trial division is trivial.
    
    def compute_phi(n):
        if n == 1:
            return 1
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result
    
    # Search for the minimal n
    for n in range(1, 1000001):
        if compute_phi(n) % 120 == 0:
            return n
        
        # If we've gone past a reasonable small number, we might switch to a sieve
        # but the knowledge says the minimal n is small. 
        # Let's check time occasionally just in case, though it shouldn't be needed.
        if n % 1000 == 0 and time.time() > deadline:
            # Fallback: if we haven't found it (which shouldn't happen), 
            # we could try a sieve, but for now, we just continue or return a default?
            # The task says "Minimize n". If we time out, we should return the best found?
            # But we haven't found any yet in the loop if we're here.
            # Actually, if n is small, this loop finishes instantly.
            pass
            
    return 1  # Should never reach here for valid range if a solution exists

# EVOLVE-BLOCK-END
