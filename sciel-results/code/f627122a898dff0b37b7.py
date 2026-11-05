import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
        
    primes = get_primes(1000)
    n_primes = len(primes)
    
    L_max = 200 # Start smaller to be faster, expand if needed
    
    def eval_state(residues, L_max):
        counts = np.zeros(L_max + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start_x = r if r > 0 else p
            if start_x <= L_max:
                counts[start_x : L_max + 1 : p] += 1
        valid = counts[1:L_max+1] >= 2
        if np.all(valid):
            L = L_max
        else:
            L = np.argmax(~valid)
        return L, counts

    def update_counts(residues, counts, i, old_r, new_r, L_max):
        p = primes[i]
        start_x = old_r if old_r > 0 else p
        if start_x <= L_max:
            counts[start_x : L_max + 1 : p] -= 1
        start_x = new_r if new_r > 0 else p
        if start_x <= L_max:
            counts[start_x : L_max + 1 : p] += 1

    def find_L(counts, L_max):
        valid = counts[1:L_max+1] >= 2
        if np.all(valid):
            return L_max
        else:
            return np.argmax(~valid)

    best_L = 0
    best_residues = None
    best_counts = None
    
    # Initialization
    inits = [
        [1] * n_primes,
        [i % p for i, p in enumerate(primes)],
        [random.randint(0, p-1) for p in primes],
        [(i+1) % p for i, p in enumerate(primes)]
    ]
    
    for init in inits:
        L, counts = eval_state(init, L_max)
        if L > best_L:
            best_L = L
            best_residues = init.copy()
            best_counts = counts.copy()
            
    current_residues = best_residues.copy()
    current_counts = best_counts.copy()
    current_L = best_L
    
    stuck_counter = 0
    
    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = current_residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        update_counts(current_residues, current_counts, i, old_r, new_r, L_max)
        new_L = find_L(current_counts, L_max)
        
        if new_L > current_L:
            current_L = new_L
            stuck_counter = 0
            if current_L > best_L:
                best_L = current_L
                best_residues = current_residues.copy()
                best_counts = current_counts.copy()
                
                # Expand L_max if we hit the boundary
                if best_L == L_max:
                    L_max *= 2
                    # Re-evaluate best state with new L_max
                    best_L, best_counts = eval_state(best_residues, L_max)
                    current_residues = best_residues.copy()
                    current_counts = best_counts.copy()
                    current_L = best_L
        else:
            current_residues[i] = old_r
            update_counts(current_residues, current_counts, i, new_r, old_r, L_max)
            stuck_counter += 1
            
            if stuck_counter > 2000:
                # Reset to best
                current_residues = best_residues.copy()
                current_counts = best_counts.copy()
                current_L = best_L
                stuck_counter = 0

    return best_residues
