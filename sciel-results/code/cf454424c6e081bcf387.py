import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)

    # We will search for L up to M.
    M = 5000
    counts = np.zeros(M + 1, dtype=np.int64)

    # Initialize residues with a "repair-first" strategy:
    # Start with all residues = 0, then do a quick repair pass to cover [1, M]
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Quick initialization: try to cover [1, M] with at least 2 hits
    # We'll use a greedy approach for initialization
    # For each x in 1..M, if count[x] < 2, find a prime that doesn't cover x yet and set it to cover x
    # This is O(M * n_primes) which is 5000 * 168 = 840k ops, fine.
    
    # Actually, let's just start random and let SA do the work, but with a better initial state.
    # Let's try to make the first M integers covered by at least 2 residues.
    # A simple heuristic: for small primes, set residues to 0, 1, etc.
    
    # Better initialization: 
    # For each prime p, choose r_p such that it helps cover the gaps.
    # Let's just do random init but with a quick local search to get a decent start.
    
    # Random init
    for i, p in enumerate(primes):
        residues[i] = random.randint(0, p - 1)
        counts[residues[i]::p] += 1

    def calc_L(counts, M):
        # Returns the largest L such that counts[1..L] >= 2
        if counts[1] < 2:
            return 0
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()

    # Helper to extend M
    def extend_M(old_M, new_M):
        nonlocal counts, M
        old_counts = counts
        counts = np.zeros(new_M + 1, dtype=np.int64)
        # Copy old part
        counts[:old_M+1] = old_counts
        # Add contributions for new part
        for j, pj in enumerate(primes):
            rj = residues[j]
            start_m = max(0, (old_M + 1 - rj + pj - 1) // pj)
            first_idx = rj + start_m * pj
            if first_idx <= new_M:
                counts[first_idx::pj] += 1
        M = new_M

    # Quick repair phase: try to get L as high as possible quickly
    # We'll do a few rounds of targeted moves
    for _ in range(10000):
        if time.time() - start_time > deadline - 100:
            break
        if current_L >= M:
            extend_M(M, M * 2)
            current_L = calc_L(counts, M)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            continue
        
        x_target = current_L + 1
        # Find primes that don't cover x_target
        # We need to change some residue to cover x_target
        # Let's pick a random prime and set its residue to x_target % p
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        r_new = x_target % p
        
        if r_new == r_old:
            continue
            
        counts[r_old::p] -= 1
        counts[r_new::p] += 1
        
        new_L = calc_L(counts, M)
        if new_L >= current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            counts[r_new::p] -= 1
            counts[r_old::p] += 1

    # Now do Simulated Annealing for the rest of the time
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9999
    
    while time.time() - start_time < deadline:
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        # Determine target residue
        if current_L < M:
            x_target = current_L + 1
            r_target = x_target % p
        else:
            r_target = random.randint(0, p - 1)
        
        # With probability 0.8, pick r_target, else random
        if random.random() < 0.8:
            r_new = r_target
        else:
            r_new = random.randint(0, p - 1)
            
        if r_new == r_old:
            continue
            
        # Apply move
        counts[r_old::p] -= 1
        counts[r_new::p] += 1
        
        new_L = calc_L(counts, M)
        
        if new_L >= current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                # Extend M if we are close to the limit
                if best_L >= M - 50:
                    extend_M(M, M * 2)
                    current_L = calc_L(counts, M)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
        else:
            # Simulated Annealing acceptance
            delta = current_L - new_L
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_L = new_L
                residues[i] = r_new
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
        
        # Cool down
        T = max(T_min, T * cooling_rate)

    return [int(r) for r in best_residues]
