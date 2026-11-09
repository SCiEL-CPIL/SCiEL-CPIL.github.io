import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

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
    M = 100000

    def init_residues(seed_val):
        rng = random.Random(seed_val)
        residues = np.array([rng.randint(0, p - 1) for p in primes], dtype=np.int64)
        return residues

    def build_counts(residues):
        counts = np.zeros(M + 1, dtype=np.int64)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
        return counts

    def calc_initial_L(counts):
        if counts[1] < 2:
            return 0
        mask = ~(counts[1:] >= 2)
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx

    def update_L(counts, current_L, M):
        while current_L < M and counts[current_L + 1] >= 2:
            current_L += 1
        return current_L

    # Main logic
    rng_main = random.Random(42)
    residues = init_residues(42)
    counts = build_counts(residues)
    current_L = calc_initial_L(counts)
    
    best_L = current_L
    best_residues = residues.copy()

    T = 10.0
    T_min = 0.1
    cooling_rate = 0.99999
    
    step_count = 0
    while time.time() < deadline:
        # Pick a random prime
        i = rng_main.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        # Pick a new random residue
        r_new = rng_main.randint(0, p - 1)
        
        if r_new == r_old:
            continue
            
        # Apply change
        old_indices = np.arange(r_old, M + 1, p)
        new_indices = np.arange(r_new, M + 1, p)
        
        counts[old_indices] -= 1
        counts[new_indices] += 1
        residues[i] = r_new
        
        # Determine new_L
        # Check if any index <= current_L in old_indices is now < 2
        risky_indices = old_indices[old_indices <= current_L]
        
        if len(risky_indices) > 0:
            bad_mask = counts[risky_indices] < 2
            if np.any(bad_mask):
                # Find the first bad index
                first_bad_idx = risky_indices[bad_mask][0]
                new_L = first_bad_idx - 1
            else:
                # No gap in <= current_L, try to extend
                new_L = update_L(counts, current_L, M)
        else:
            # No risky indices, try to extend
            new_L = update_L(counts, current_L, M)
        
        delta = current_L - new_L
        
        if new_L >= current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            if delta > 1000:
                # Revert
                counts[new_indices] -= 1
                counts[old_indices] += 1
                residues[i] = r_old
            else:
                prob = np.exp(-delta / T)
                if rng_main.random() < prob:
                    current_L = new_L
                else:
                    # Revert
                    counts[new_indices] -= 1
                    counts[old_indices] += 1
                    residues[i] = r_old
        
        T = max(T_min, T * cooling_rate)
        
        step_count += 1

    return [int(r) for r in best_residues]
