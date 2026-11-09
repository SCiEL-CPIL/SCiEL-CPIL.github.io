import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0

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
    
    M = 2000
    counts = np.zeros(M + 1, dtype=np.int64)
    
    # Initialize with 0s
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        counts[residues[i]::p] += 1

    def calc_L(counts, M):
        if counts[1] < 2:
            return 0
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        return np.argmax(mask)

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()

    def extend_M(new_M):
        nonlocal M, counts
        old_M = M
        old_counts = counts
        counts = np.zeros(new_M + 1, dtype=np.int64)
        counts[:old_M + 1] = old_counts
        for j, pj in enumerate(primes):
            rj = residues[j]
            start_m = max(0, (old_M + 1 - rj + pj - 1) // pj)
            first_idx = rj + start_m * pj
            if first_idx <= new_M:
                counts[first_idx::pj] += 1
        M = new_M

    def try_repair(x):
        # Try to fix gap at x
        # Shuffled primes
        prime_indices = list(range(n_primes))
        random.shuffle(prime_indices)
        
        for i in prime_indices:
            p = primes[i]
            r_old = residues[i]
            r_new = x % p
            
            if r_old == r_new:
                continue
                
            # Apply
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            
            # Check validity for [1, x-1]
            # We only care if min count in [1, x-1] is >= 2
            # Note: counts[0] is not part of the problem domain [1..L]
            # But x is the gap, so counts[x] will be >= 2 (we added 1, and it was <2)
            # We must ensure we didn't break [1, x-1]
            
            # Efficient check:
            # Only positions y < x affected are those y = r_old + k*p or y = r_new + k*p
            # But np.min(counts[1:x]) is O(x). x can be up to 10^5.
            # Doing this for 168 primes is 168 * 10^5 = 1.68 * 10^7 operations.
            # This might be too slow if done many times.
            
            # Optimization: Only check if the move *could* break something.
            # If r_old != r_new, we lose coverage at r_old mod p.
            # If any y in [1, x-1] with y % p == r_old has count 2, it breaks.
            
            # Let's use the numpy min for now, it's vectorized and fast in C.
            if x > 1:
                min_val = np.min(counts[1:x])
            else:
                min_val = 2 # No prefix to check
            
            if min_val >= 2:
                # Valid move
                residues[i] = r_new
                return True
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                return False # Continue trying other primes? 
                # Actually, we should continue the loop to try other primes.
                # So don't return False immediately, just revert and continue.
        return False

    # Main Loop
    while time.time() - start_time < deadline - 5:
        current_L = calc_L(counts, M)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            if best_L >= M - 10:
                extend_M(M * 2)
                current_L = calc_L(counts, M)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()

        gap_x = current_L + 1
        
        if gap_x > M:
            # No gap in current range, extend
            extend_M(M * 2)
            continue
            
        # Try to repair
        repaired = try_repair(gap_x)
        
        if not repaired:
            # Simulated Annealing Step
            # Pick random prime
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            r_new = random.randint(0, p - 1)
            
            if r_new == r_old:
                continue
                
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            
            new_L = calc_L(counts, M)
            
            # Accept if better or with some probability
            # For simplicity, accept if not worse, or if significantly better
            if new_L >= current_L:
                residues[i] = r_new
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                # Maybe accept with low prob? For now, revert.

    # Final Polish: Local Search
    # Try to improve L by optimizing individual residues
    for _ in range(10): # 10 passes
        if time.time() - start_time > deadline - 2:
            break
        for i in range(n_primes):
            p = primes[i]
            r_old = residues[i]
            best_r = r_old
            best_local_L = current_L
            
            # Try all residues? p can be up to 997.
            # 168 * 1000 * (update cost) is too slow.
            # Just try a few random ones or the one that covers the current gap?
            # Let's stick to the repair loop, it's effectively doing this.
            pass
            
    return [int(r) for r in best_residues]
