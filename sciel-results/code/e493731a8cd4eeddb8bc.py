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
    
    # Precompute X and ModMatrix for speed
    max_L = 100000
    X = np.arange(1, max_L + 1)
    ModMatrix = np.stack([X % p for p in primes]) # Shape: (168, 100000)
        
    def score(residues):
        r_arr = np.array(residues)
        # Compare each prime's mod values against its residue
        # r_arr[:, None] has shape (168, 1)
        # ModMatrix has shape (168, 100000)
        # Result is (168, 100000) boolean
        matches = ModMatrix == r_arr[:, None]
        counts = np.sum(matches, axis=0)
        bad = np.where(counts < 2)[0]
        if len(bad) > 0:
            return bad[0]
        return max_L

    # Initial residues: Random
    residues = [random.randint(0, p - 1) for p in primes]
    
    current_score = score(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    start_time = time.time()
    # Safety margin: 110s
    deadline = start_time + 110
    
    # Phase 1: Greedy Repair
    repair_deadline = start_time + 5
    while time.time() < repair_deadline:
        L = current_score
        if L >= max_L:
            break
        K = L + 1
        # Find primes that do NOT cover K currently
        candidates = []
        for i, p in enumerate(primes):
            if residues[i] != K % p:
                candidates.append(i)
        
        if not candidates:
            break
        
        # Pick a random candidate
        i = random.choice(candidates)
        old_r = residues[i]
        residues[i] = K % primes[i]
        
        new_score = score(residues)
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
        else:
            residues[i] = old_r

    # Phase 2: Hill Climbing
    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        old_r = residues[i]
        new_r = random.randint(0, primes[i] - 1)
        
        residues[i] = new_r
        new_score = score(residues)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
        else:
            residues[i] = old_r
            
    return best_residues
