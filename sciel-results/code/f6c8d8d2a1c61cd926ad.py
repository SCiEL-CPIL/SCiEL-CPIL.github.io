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
    
    # Smart initialization: for each prime, pick residue that maximizes coverage of [1, M]
    # This is a greedy approach: for each prime, pick the residue that covers the most numbers
    # that are currently under-covered (count < 2).
    
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # First pass: random initialization
    for i, p in enumerate(primes):
        residues[i] = random.randint(0, p - 1)
        counts[residues[i]::p] += 1
    
    # Second pass: greedy improvement - for each prime, try all residues and keep the best
    for i in range(n_primes):
        p = primes[i]
        r_old = residues[i]
        best_r = r_old
        best_score = -1
        for r in range(p):
            if r == r_old:
                continue
            # Tentatively apply
            counts[r_old::p] -= 1
            counts[r::p] += 1
            # Score: number of positions in [1, M] with count >= 2
            score = np.sum(counts[1:M+1] >= 2)
            if score > best_score:
                best_score = score
                best_r = r
            # Revert
            counts[r::p] -= 1
            counts[r_old::p] += 1
        if best_r != r_old:
            residues[i] = best_r
            counts[r_old::p] -= 1
            counts[best_r::p] += 1
    
    def calc_L(counts, M):
        # Returns the largest L such that counts[1..L] >= 2
        if counts[1] < 2:
            return 0
        mask = counts[1:M+1] < 2
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx
    
    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()
    
    def is_valid_move(counts, i, r_old, r_new, L, p):
        if r_old == r_new:
            return True
        start_y = r_old
        if start_y == 0:
            start_y = p
        for y in range(start_y, L + 1, p):
            if counts[y] == 2:
                return False
        return True
    
    def apply_move(counts, i, r_old, r_new, p):
        if r_old != r_new:
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
    
    def revert_move(counts, i, r_old, r_new, p):
        if r_old != r_new:
            counts[r_new::p] -= 1
            counts[r_old::p] += 1
    
    def extend_counts(counts, M, new_M, primes, residues):
        old_counts = counts
        counts = np.zeros(new_M + 1, dtype=np.int64)
        counts[:M+1] = old_counts
        for j, pj in enumerate(primes):
            rj = residues[j]
            start_m = max(0, (M + 1 - rj + pj - 1) // pj)
            first_idx = rj + start_m * pj
            if first_idx <= new_M:
                counts[first_idx::pj] += 1
        return counts
    
    # MCMC with Metropolis-Hastings
    T = 50.0
    T_min = 1.0
    cooling_rate = 0.999
    
    # Energy function: penalize first uncovered position and deficit depth
    def energy(L, M):
        # Lower is better
        # Penalize how far L is from M
        return M - L
    
    stuck_counter = 0
    
    while time.time() - start_time < deadline:
        # 1. Try to repair the gap
        gap_x = current_L + 1
        
        if gap_x > M:
            # Extend M
            new_M = M * 2
            counts = extend_counts(counts, M, new_M, primes, residues)
            M = new_M
            current_L = calc_L(counts, M)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            continue
        
        found_valid_move = False
        
        # Try targeted moves to fix the gap
        prime_indices = list(range(n_primes))
        random.shuffle(prime_indices)
        
        for i in prime_indices:
            p = primes[i]
            r_old = residues[i]
            
            if r_old == gap_x % p:
                continue
                
            r_new = gap_x % p
            
            if is_valid_move(counts, i, r_old, r_new, current_L, p):
                apply_move(counts, i, r_old, r_new, p)
                residues[i] = r_new
                
                new_L = calc_L(counts, M)
                
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    if new_L >= M - 50:
                        new_M = M * 2
                        counts = extend_counts(counts, M, new_M, primes, residues)
                        M = new_M
                        current_L = calc_L(counts, M)
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
                else:
                    current_L = new_L
                
                found_valid_move = True
                stuck_counter = 0
                break
        
        if found_valid_move:
            continue
        
        # 2. No valid repair move found. Use Metropolis-Hastings.
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        r_new = random.randint(0, p - 1)
        
        if r_new == r_old:
            continue
        
        apply_move(counts, i, r_old, r_new, p)
        
        new_L = calc_L(counts, M)
        
        old_energy = energy(current_L, M)
        new_energy = energy(new_L, M)
        
        delta_E = new_energy - old_energy
        
        if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                if best_L >= M - 50:
                    new_M = M * 2
                    counts = extend_counts(counts, M, new_M, primes, residues)
                    M = new_M
                    current_L = calc_L(counts, M)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
            stuck_counter = 0
        else:
            revert_move(counts, i, r_old, r_new, p)
            stuck_counter += 1
        
        T = max(T_min, T * cooling_rate)
        
        # If stuck, do a local search: try all residues for a random prime
        if stuck_counter > 100:
            stuck_counter = 0
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            best_r = r_old
            best_L_local = current_L
            for r in range(p):
                if r == r_old:
                    continue
                counts[r_old::p] -= 1
                counts[r::p] += 1
                L_local = calc_L(counts, M)
                if L_local > best_L_local:
                    best_L_local = L_local
                    best_r = r
                counts[r::p] -= 1
                counts[r_old::p] += 1
            if best_r != r_old:
                residues[i] = best_r
                counts[r_old::p] -= 1
                counts[best_r::p] += 1
                current_L = best_L_local
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                    if best_L >= M - 50:
                        new_M = M * 2
                        counts = extend_counts(counts, M, new_M, primes, residues)
                        M = new_M
                        current_L = calc_L(counts, M)
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()

    return [int(r) for r in best_residues]
