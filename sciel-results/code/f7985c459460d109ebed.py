import numpy as np
import time

def construct():
    def primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = primes_upto(1000)
    n_primes = len(primes)
    
    # We want to maximize L. 
    # Let's set MAX_CHECK to 20000 to be safe, but scoring becomes slower.
    # 20000 * 168 = 3.36M ops. Still fast in NumPy.
    MAX_L = 20000
    x_vals = np.arange(1, MAX_L + 1)
    mod_matrix = np.zeros((MAX_L, n_primes), dtype=np.uint16)
    for j, p in enumerate(primes):
        mod_matrix[:, j] = x_vals % p
        
    def get_score(residues):
        # residues: (n_primes,)
        # counts: (MAX_L,)
        counts = (mod_matrix == residues[None, :]).sum(axis=1)
        # Find first index where count < 2
        # Using np.argmax on a boolean mask where condition is False
        # If all are True, argmax returns 0 (which is wrong if we expect MAX_L)
        # Better:
        bad = counts < 2
        if not np.any(bad):
            return MAX_L
        return int(np.argmax(bad))
        
    # Initial residues
    residues = np.array([np.random.randint(p) for p in primes], dtype=np.uint16)
    current_score = get_score(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110.0
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            residues = np.array([np.random.randint(p) for p in primes], dtype=np.uint16)
            current_score = get_score(residues)
            if current_score > best_score:
                best_residues = residues.copy()
                best_score = current_score
            no_improve_count = 0
            continue
            
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)
        if new_r == old_r:
            continue
            
        residues[i] = new_r
        new_score = get_score(residues)
        
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
        else:
            residues[i] = old_r
            no_improve_count += 1
            
    return list(best_residues)
