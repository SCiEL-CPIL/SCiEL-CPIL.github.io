import time
import numpy as np
import random

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _build_counts(residues, primes, max_L):
    """Build coverage count array for positions 1..max_L."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = p if r == 0 else r
        if start <= max_L:
            pos = np.arange(start, max_L + 1, p)
            if len(pos) > 0:
                counts[pos] += 1
    return counts

def _get_L(counts, max_L):
    """Find largest L such that all positions 1..L have count >= 2."""
    # We care about 1..max_L
    # Find first index in 1..max_L where count < 2
    deficient = np.where(counts[1:max_L+1] < 2)[0]
    if len(deficient) == 0:
        return max_L
    return int(deficient[0])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # 1. Initial Solution: Greedy Forward Construction
    residues = [0] * n_primes
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    # Greedy init on a small window
    window = 2000
    temp_counts = np.zeros(window + 1, dtype=np.int32)
    
    for i in range(n_primes):
        p = primes[i]
        best_r = 0
        # Find deficient positions in 1..window
        def_pos = np.where(temp_counts[1:window+1] < 2)[0] + 1
        if len(def_pos) == 0:
            residues[i] = 0
        else:
            # Vote for residues covering deficient positions
            votes = {}
            for pos in def_pos[:50]:
                r_needed = pos % p
                if r_needed not in votes:
                    votes[r_needed] = 0
                votes[r_needed] += 1
            if votes:
                best_r = max(votes, key=votes.get)
            else:
                best_r = 0
                
        residues[i] = best_r
        
        # Update temp_counts
        start = p if best_r == 0 else best_r
        if start <= window:
            pos = np.arange(start, window + 1, p)
            if len(pos) > 0:
                temp_counts[pos] += 1

    # Build full counts and get initial L
    counts = _build_counts(residues, primes, max_L)
    current_L = _get_L(counts, max_L)
    best_L = current_L
    best_residues = residues[:]
    
    no_improve_count = 0
    max_no_improve = 20 # Passes without improvement before restart
    
    while (time.time() - start_time) < deadline:
        improved_in_pass = False
        order = list(range(n_primes))
        random.shuffle(order)
        
        for i in order:
            if (time.time() - start_time) > deadline - 2:
                break
                
            p = primes[i]
            current_r = residues[i]
            
            # Determine candidates
            if p < 100:
                candidates = list(range(p))
            else:
                candidates = [current_r]
                # Sample 50 random residues
                candidates += list(np.random.choice(p, size=min(p-1, 50), replace=False))
                candidates = list(set(candidates))
            
            best_new_r = current_r
            best_new_L = current_L
            
            # Precompute old positions
            old_start = p if current_r == 0 else current_r
            if old_start <= max_L:
                old_pos = np.arange(old_start, max_L + 1, p)
            else:
                old_pos = np.array([], dtype=np.int32)
            
            for new_r in candidates:
                if new_r == current_r:
                    continue
                
                # Apply change
                if len(old_pos) > 0:
                    counts[old_pos] -= 1
                
                new_start = p if new_r == 0 else new_r
                if new_start <= max_L:
                    new_pos = np.arange(new_start, max_L + 1, p)
                    if len(new_pos) > 0:
                        counts[new_pos] += 1
                
                # Evaluate L
                new_L = _get_L(counts, max_L)
                
                # Revert change
                if len(old_pos) > 0:
                    counts[old_pos] += 1
                if new_start <= max_L:
                    new_pos = np.arange(new_start, max_L + 1, p)
                    if len(new_pos) > 0:
                        counts[new_pos] -= 1
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_r = new_r
            
            if best_new_r != current_r:
                # Commit
                residues[i] = best_new_r
                if len(old_pos) > 0:
                    counts[old_pos] -= 1
                new_start = p if best_new_r == 0 else best_new_r
                if new_start <= max_L:
                    new_pos = np.arange(new_start, max_L + 1, p)
                    if len(new_pos) > 0:
                        counts[new_pos] += 1
                
                current_L = best_new_L
                improved_in_pass = True
                no_improve_count = 0
                
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                
        if not improved_in_pass:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart
                num_perturb = random.randint(2, 5)
                indices = random.sample(range(n_primes), num_perturb)
                for idx in indices:
                    p = primes[idx]
                    old_r = residues[idx]
                    new_r = random.randint(0, p-1)
                    residues[idx] = new_r
                    
                    old_start = p if old_r == 0 else old_r
                    if old_start <= max_L:
                        old_pos = np.arange(old_start, max_L + 1, p)
                        if len(old_pos) > 0:
                            counts[old_pos] -= 1
                    
                    new_start = p if new_r == 0 else new_r
                    if new_start <= max_L:
                        new_pos = np.arange(new_start, max_L + 1, p)
                        if len(new_pos) > 0:
                            counts[new_pos] += 1
                
                current_L = _get_L(counts, max_L)
                no_improve_count = 0
                
                # If we dropped significantly, revert to best
                if current_L < best_L - 10:
                    residues = best_residues[:]
                    counts = _build_counts(residues, primes, max_L)
                    current_L = best_L

    return best_residues
