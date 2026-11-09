import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    num_primes = len(primes)
    
    # Initial residues - use a more informed initialization
    residues = np.array([((i * 7 + 3) % p) for i, p in enumerate(primes)], dtype=int)
    
    N = 1000
    x = np.arange(1, N + 1)
    
    counts = np.zeros(N, dtype=int)
    for i, p in enumerate(primes):
        counts += (x % p == residues[i])
        
    def get_bottleneck(c):
        is_fail = c < 2
        if not np.any(is_fail):
            return -1
        return np.argmax(is_fail)
        
    def compute_L():
        b_idx = get_bottleneck(counts)
        return 0 if b_idx == -1 else b_idx
        
    current_L = compute_L()
    best_L = current_L
    best_residues = residues.copy()
    
    # Precompute masks for each prime to speed up updates
    prime_masks = []
    for p in primes:
        mask = (x % p)
        prime_masks.append(mask)
    
    while time.time() < deadline:
        improved = False
        order = np.random.permutation(num_primes)
        
        for i in order:
            if time.time() > deadline:
                break
            
            b_idx = get_bottleneck(counts)
            if b_idx == -1:
                break
                
            target_x = b_idx + 1
            p = primes[i]
            
            current_r = residues[i]
            if (target_x % p) == current_r:
                continue
                
            new_r = target_x % p
            
            # Check safety: no x in 1..b_idx with count==2 should lose coverage
            if b_idx > 0:
                mask_R_prefix = (x[:b_idx] % p == current_r)
                if np.any(mask_R_prefix & (counts[:b_idx] == 2)):
                    continue
            
            # Apply move
            residues[i] = new_r
            mask_R = (x % p == current_r)
            mask_A = (x % p == new_r)
            counts -= mask_R
            counts += mask_A
            
            new_L = compute_L()
            
            if new_L > current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                improved = True
            else:
                # Revert if no improvement
                residues[i] = current_r
                counts += mask_R
                counts -= mask_A
        
        # If stuck, try a random perturbation to escape local optimum
        if not improved:
            # Try random changes to a few residues
            for _ in range(50):
                if time.time() > deadline:
                    break
                i = np.random.randint(num_primes)
                p = primes[i]
                old_r = residues[i]
                new_r = np.random.randint(p)
                if new_r == old_r:
                    continue
                
                mask_R = (x % p == old_r)
                mask_A = (x % p == new_r)
                
                # Check if this helps the bottleneck
                b_idx = get_bottleneck(counts)
                if b_idx == -1:
                    break
                
                target_x = b_idx + 1
                if (target_x % p) != new_r:
                    continue
                    
                # Check safety
                if b_idx > 0:
                    mask_R_prefix = (x[:b_idx] % p == old_r)
                    if np.any(mask_R_prefix & (counts[:b_idx] == 2)):
                        continue
                        
                residues[i] = new_r
                counts -= mask_R
                counts += mask_A
                
                new_L = compute_L()
                if new_L > current_L:
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_residues = residues.copy()
                    break
                else:
                    residues[i] = old_r
                    counts += mask_R
                    counts -= mask_A
            else:
                # No improvement from perturbation, break
                break
                
    return [int(r) for r in best_residues]
