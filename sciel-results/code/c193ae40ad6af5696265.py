"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 80.0  # 80s safety margin within 90s limit
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Return all primes in 2..256
    result = [a for a in range(2, 257) if is_prime(a)]
    
    # Verify the constraint: for all distinct a,b,c in result, a does not divide b*c
    # This is a sanity check; for primes it should always hold
    n = len(result)
    for i in range(n):
        a = result[i]
        for j in range(n):
            if j == i:
                continue
            b = result[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = result[k]
                if (b * c) % a == 0:
                    # This should never happen for distinct primes
                    pass
    
    return result
# EVOLVE-BLOCK-END
