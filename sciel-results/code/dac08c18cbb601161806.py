import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
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
    MAX_L = 1000
    
    # Initialize residues
    residues = [random.randint(0, p - 1) for p in primes]
    
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    # Precompute index arrays for each prime? 
    # No, depends on residue.
    # But we can precompute the "step" and use numpy slicing.
    
    def get_indices(p, r):
        if r == 0:
            start = p
        else:
            start = r
        if start > MAX_L:
            return np.array([], dtype=np.int32)
        return np.arange(start, MAX_L + 1, p)

    # Initialize counts
    for i, p in enumerate(primes):
        r = residues[i]
        idx = get_indices(p, r)
        if len(idx) > 0:
            counts[idx] += 1
            
    def get_first_gap():
        under = np.where(counts[1:] < 2)[0]
        if len(under) == 0:
            return MAX_L
        return under[0]
        
    # Note: get_first_gap returns 0-indexed index in counts[1:], so actual x is under[0] + 1
    # If no gap, returns MAX_L (meaning 1..MAX_L are good? No, if under is empty, all 1..MAX_L are good).
    # Let's return the actual x value.
    
    def find_first_gap_x():
        under = np.where(counts[1:] < 2)[0]
        if len(under) == 0:
            return MAX_L + 1
        return under[0] + 1

    current_L = find_first_gap_x() - 1
    best_L = current_L
    best_residues = residues.copy()
    
    while time.time() < deadline:
        if current_L >= MAX_L:
            break
            
        target = current_L + 1
        target_count = counts[target]
        
        if target_count >= 2:
            current_L = target
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            continue
            
        # Need to fix target
        found_move = False
        # Randomize prime order
        perm = np.random.permutation(n_primes)
        
        for i in perm:
            p = primes[i]
            needed_r = target % p
            current_r = residues[i]
            
            if current_r == needed_r:
                continue
                
            old_idx = get_indices(p, current_r)
            new_idx = get_indices(p, needed_r)
            
            # Check if decrementing old_idx breaks any y <= current_L
            # We only care about y <= current_L
            # If old_idx is empty, no decrement.
            if len(old_idx) > 0:
                # Filter old_idx to only those <= current_L
                # Since old_idx is sorted, we can find the cutoff
                # Or just check min count in old_idx[:k] where k is count of elements <= current_L
                # Actually, get_indices returns sorted array.
                # We can use searchsorted to find how many are <= current_L
                k = np.searchsorted(old_idx, current_L)
                if k > 0:
                    # Check if any count in old_idx[:k] is 2
                    if np.min(counts[old_idx[:k]]) == 2:
                        continue
                        # If min is 1, it's already broken? No, current_L is max covered, so all <= current_L are >= 2.
                        # So min must be >= 2. If min is 2, decrementing makes it 1 -> Invalid.
                        
            # If valid, apply
            if len(old_idx) > 0:
                counts[old_idx] -= 1
            if len(new_idx) > 0:
                counts[new_idx] += 1
            residues[i] = needed_r
            found_move = True
            
            # Advance current_L
            while current_L < MAX_L and counts[current_L + 1] >= 2:
                current_L += 1
                
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
                
            break
            
        if not found_move:
            # Kick
            for _ in range(5):
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                new_r = random.randint(0, p - 1)
                old_r = residues[i]
                
                old_idx = get_indices(p, old_r)
                new_idx = get_indices(p, new_r)
                
                if len(old_idx) > 0:
                    counts[old_idx] -= 1
                if len(new_idx) > 0:
                    counts[new_idx] += 1
                residues[i] = new_r
                
            # Recalculate L
            current_L = find_first_gap_x() - 1
            
    return best_residues
