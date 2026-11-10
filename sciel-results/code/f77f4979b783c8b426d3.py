import time
import random
import numpy as np

def construct():
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Seed residues
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Precompute X for scoring
    max_L = 100000
    X_full = np.arange(1, max_L + 1)
    
    def score(residues):
        # Check in chunks of 1000
        window_size = 1000
        for start in range(0, max_L, window_size):
            end = min(start + window_size, max_L)
            X = X_full[start:end]
            counts = np.zeros(len(X), dtype=np.int8)
            for i, p in enumerate(primes):
                r = residues[i]
                counts += (X % p == r)
            
            bad_indices = np.where(counts < 2)[0]
            if len(bad_indices) > 0:
                # start is 0-based index in X_full
                # X_full[start] is the integer value
                # The integer value is X_full[start] + bad_indices[0]
                # The score is the number of integers covered, which is (first_bad_integer - 1)
                # first_bad_integer = X_full[start] + bad_indices[0]
                # So score = X_full[start] + bad_indices[0] - 1
                return X_full[start] + bad_indices[0] - 1
        return max_L

    current_residues = residues.copy()
    current_score = score(current_residues)
    best_residues = current_residues.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + 140
    
    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        old_r = current_residues[i]
        new_r = random.randint(0, primes[i] - 1)
        
        current_residues[i] = new_r
        new_score = score(current_residues)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues.copy()
        else:
            current_residues[i] = old_r
            
    return best_residues
