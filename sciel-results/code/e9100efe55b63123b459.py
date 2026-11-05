import time
import random
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Search horizon
    H = 5000
    
    # Initialize residues
    residues = [random.randint(0, p - 1) for p in primes]
    
    # counts[x] for x in 0..H
    counts = np.zeros(H + 1, dtype=np.int32)
    
    def apply_change(idx, old_r, new_r):
        p = primes[idx]
        # Remove old
        start_old = old_r if old_r > 0 else p
        if start_old <= H:
            counts[start_old::p] -= 1
        # Add new
        start_new = new_r if new_r > 0 else p
        if start_new <= H:
            counts[start_new::p] += 1

    def find_L():
        # Find first x in 1..H where counts[x] < 2
        # Using numpy
        below = np.where(counts[1:H+1] < 2)[0]
        if len(below) == 0:
            return H
        return below[0] + 1

    # Initial setup
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= H:
            counts[start::p] += 1
            
    current_L = find_L()
    best_L = current_L
    best_residues = residues[:]
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Perform a random kick to escape local minimum
            # Change 3 random residues
            for _ in range(3):
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                new_r = random.randint(0, p - 1)
                if new_r != residues[idx]:
                    apply_change(idx, residues[idx], new_r)
                    residues[idx] = new_r
            current_L = find_L()
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
            no_improve_count = 0
            continue

        # 60% Directed, 40% Random
        if random.random() < 0.6 and current_L < H:
            x = current_L
            # Find a prime that does not cover x
            found_idx = -1
            for _ in range(5):
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                if x % p != residues[idx]:
                    found_idx = idx
                    break
            
            if found_idx != -1:
                idx = found_idx
                p = primes[idx]
                new_r = x % p
                old_r = residues[idx]
                
                apply_change(idx, old_r, new_r)
                residues[idx] = new_r
                new_L = find_L()
                
                if new_L > current_L:
                    current_L = new_L
                    no_improve_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                else:
                    apply_change(idx, new_r, old_r)
                    residues[idx] = old_r
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Random move
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            apply_change(idx, old_r, new_r)
            residues[idx] = new_r
            new_L = find_L()
            
            if new_L > current_L:
                current_L = new_L
                no_improve_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
            else:
                apply_change(idx, new_r, old_r)
                residues[idx] = old_r
                no_improve_count += 1

    return best_residues
