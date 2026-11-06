import time
import numpy as np

def construct():
    # Generate first 168 primes
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = []
    n = 2
    while len(primes) < 168:
        if is_prime(n):
            primes.append(n)
        n += 1

    # Maximum L to consider.
    MAX_L = 5000
    x = np.arange(1, MAX_L + 1)
    # coverage[i] corresponds to integer x[i] = i + 1
    coverage = np.zeros(MAX_L, dtype=np.int32)
    
    current_L = 0
    residues = []
    
    start_time = time.time()
    deadline = start_time + 110.0

    for p in primes:
        if time.time() > deadline:
            break
            
        best_r = 0
        best_L = current_L
        
        for r in range(p):
            mask = (x % p == r)
            temp_cov = coverage + mask
            
            # Check from current_L onwards
            # Index for integer current_L + 1 is current_L
            check_slice = temp_cov[current_L:]
            
            if len(check_slice) == 0:
                L_candidate = MAX_L
            else:
                invalid_mask = check_slice < 2
                if np.any(invalid_mask):
                    first_invalid = np.argmax(invalid_mask)
                    L_candidate = current_L + first_invalid
                else:
                    L_candidate = MAX_L
                    
            if L_candidate > best_L:
                best_L = L_candidate
                best_r = r
                if best_L == MAX_L:
                    break # Can't do better than MAX_L
                    
        mask = (x % p == best_r)
        coverage += mask
        current_L = best_L
        residues.append(best_r)
        
    # Pad if necessary
    while len(residues) < 168:
        residues.append(0)
        
    return residues
