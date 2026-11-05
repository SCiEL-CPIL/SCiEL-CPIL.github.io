"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import random
import time

def construct():
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    
    # Max L to check during search
    MAX_L = 20000
    
    # Initialize counts array
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def get_L_from_counts(counts):
        # Find first x in 1..MAX_L such that counts[x] < 2
        # counts[0] is ignored
        bad = counts[1:] < 2
        if not np.any(bad):
            return MAX_L
        return np.argmax(bad)

    def update_counts_for_prime(residues, counts, primes, idx, new_r, MAX_L):
        """
        Updates counts by changing residue of prime at idx to new_r.
        Assumes counts currently reflects old residue.
        """
        p = primes[idx]
        old_r = residues[idx]
        
        if old_r == new_r:
            return

        # Remove old coverage
        start_old = old_r
        if start_old == 0:
            start_old = p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
            
        # Add new coverage
        start_new = new_r
        if start_new == 0:
            start_new = p
        if start_new <= MAX_L:
            counts[start_new::p] += 1
            
        residues[idx] = new_r

    # Initialize residues randomly
    random.seed(42)
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Initial count calculation
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start == 0:
            start = p
        if start <= MAX_L:
            counts[start::p] += 1
            
    current_L = get_L_from_counts(counts)
    best_L = current_L
    best_residues = residues[:]
    
    start_time = time.time()
    time_limit = 110.0
    
    # Hill Climbing with Restarts
    num_restarts = 10
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # If we have a good best, start from it. Else random.
        if best_L > 0:
            residues = best_residues[:]
            # Rebuild counts from best_residues
            counts[:] = 0
            for i, p in enumerate(primes):
                r = residues[i]
                start = r
                if start == 0:
                    start = p
                if start <= MAX_L:
                    counts[start::p] += 1
            current_L = get_L_from_counts(counts)
        else:
            residues = [random.randint(0, p - 1) for p in primes]
            counts[:] = 0
            for i, p in enumerate(primes):
                r = residues[i]
                start = r
                if start == 0:
                    start = p
                if start <= MAX_L:
                    counts[start::p] += 1
            current_L = get_L_from_counts(counts)
            
        # Coordinate Descent Loop
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # Shuffle prime order to avoid bias
            prime_indices = list(range(n_primes))
            random.shuffle(prime_indices)
            
            for idx in prime_indices:
                if time.time() - start_time > time_limit:
                    break
                    
                p = primes[idx]
                old_r = residues[idx]
                
                # Sample candidate residues
                # If p is small, try all. If p is large, try random sample.
                if p <= 50:
                    candidates = list(range(p))
                else:
                    candidates = random.sample(range(p), min(p, 30))
                
                # Ensure current residue is in candidates to allow "no change" comparison implicitly
                # We only accept if new_L > current_L
                best_local_r = old_r
                best_local_L = current_L
                
                for r in candidates:
                    if r == old_r:
                        continue
                    
                    # Apply change
                    update_counts_for_prime(residues, counts, primes, idx, r, MAX_L)
                    
                    # Evaluate
                    new_L = get_L_from_counts(counts)
                    
                    if new_L > best_local_L:
                        best_local_L = new_L
                        best_local_r = r
                    else:
                        # Revert immediately to keep counts clean for next candidate
                        update_counts_for_prime(residues, counts, primes, idx, old_r, MAX_L)
                
                # Apply the best found for this prime
                if best_local_r != old_r:
                    # The counts currently reflect old_r (if we reverted) or best_local_r (if it was the last one tested and accepted?)
                    # Wait, the logic above reverts if new_L <= best_local_L.
                    # If best_local_r is found, it might be the last one tested, so counts reflect best_local_r.
                    # Or it might be an earlier one, and we reverted back to old_r.
                    # Let's just force the state to be old_r before applying best_local_r to be safe?
                    # Actually, if best_local_r was found, and we didn't revert it (because it was better than the previous best_local_L),
                    # then counts reflect best_local_r.
                    # If best_local_r is old_r, counts reflect old_r.
                    # If best_local_r is some other r, and we found it, then for subsequent candidates we reverted to old_r.
                    # So currently, if best_local_r != old_r, counts reflect old_r (because the last candidate tested was likely worse and reverted, 
                    # OR the last candidate was the best and we didn't revert it? No, we only revert if new_L <= best_local_L.
                    # If r is the best, new_L > best_local_L (which was updated to new_L). So we do NOT revert.
                    # So counts reflect best_local_r.
                    
                    # So we just update the record.
                    current_L = best_local_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                    improved = True
                else:
                    # Ensure counts are consistent (they should be, as we reverted or never changed)
                    pass

    return best_residues

# EVOLVE-BLOCK-END
