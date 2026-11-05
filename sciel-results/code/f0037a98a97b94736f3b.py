import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    # Precompute positions for each prime: positions where x ≡ r (mod p) for x in 1..MAX_L
    # For prime p, residue r, positions are r, r+p, r+2p, ... up to MAX_L
    # If r == 0, positions are p, 2p, 3p, ...
    # We'll store for each prime a list of numpy arrays: pos[p_idx][r] = array of positions
    
    # Actually, for efficiency, let's just compute on the fly but use a smart approach.
    # For each prime p, the positions for residue r are:
    #   if r == 0: p, 2p, 3p, ..., floor(MAX_L/p)*p
    #   if r > 0: r, r+p, r+2p, ..., up to MAX_L
    
    # Let's precompute the "base" positions for each prime (for r=0) and then shift.
    # Actually, it's easier to just use the slice notation counts[start::p] += 1
    
    def build_counts(residues):
        """Build the full count array for given residues."""
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        return counts

    def find_L(counts):
        """Find the largest L such that counts[1..L] >= 2."""
        # Find first index >= 1 where count < 2
        bad = np.where(counts[1:MAX_L+1] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0]

    # --- Phase 1: Smart Greedy Initialization ---
    # Key insight: small primes have high frequency, so their residues matter most.
    # Process primes in order of increasing p (already sorted).
    # For each prime, pick the residue that maximizes coverage of the "frontier" - 
    # the region near the current L where coverage is deficient.
    
    def greedy_construct(window_size=20000):
        residues = [0] * n_primes
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        
        for idx in range(n_primes):
            p = primes[idx]
            best_r = 0
            best_score = -1
            
            # For small primes, try all residues; for large, sample
            if p <= 30:
                candidates = list(range(p))
            else:
                # Sample residues, biased toward covering the frontier
                candidates = list(np.random.choice(p, size=min(p, 50), replace=False))
            
            # Focus on a window around the current "frontier"
            # Current L estimate
            bad = np.where(counts[1:window_size+1] < 2)[0]
            if len(bad) == 0:
                current_L = window_size
            else:
                current_L = bad[0]
            
            # We want to cover positions in [1, current_L + some_margin]
            # Specifically, positions with count == 1 are most valuable to boost
            # And positions with count == 0 are critical
            
            for r in candidates:
                start = r if r > 0 else p
                if start > MAX_L:
                    continue
                
                # Count how many positions in the critical region we'd cover
                # Region: 1 to min(MAX_L, current_L + p*2)
                end_region = min(MAX_L, current_L + p * 3)
                pos = np.arange(start, end_region + 1, p)
                if len(pos) == 0:
                    continue
                pos = pos[pos <= MAX_L]
                if len(pos) == 0:
                    continue
                
                cur_counts = counts[pos]
                # Score: heavily weight positions with count==0, then count==1
                # Positions with count==0 becoming 1: good but not enough (need 2)
                # Positions with count==1 becoming 2: great!
                # Positions with count>=2 becoming 3: some waste but ok
                score = 0
                score += 10 * np.sum(cur_counts == 0)  # Covering holes
                score += 5 * np.sum(cur_counts == 1)   # Boosting to 2
                score += 0.1 * np.sum(cur_counts >= 2) # Slight benefit for extra coverage
                
                if score > best_score:
                    best_score = score
                    best_r = r
            
            residues[idx] = best_r
            r = best_r
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        
        L = find_L(counts)
        return residues, counts, L

    # Run multiple greedy attempts
    best_init_L = 0
    best_init_res = [0] * n_primes
    best_init_counts = None
    
    for attempt in range(5):
        res, cnt, L = greedy_construct()
        if L > best_init_L:
            best_init_L = L
            best_init_res = res
            best_init_counts = cnt

    residues = best_init_res
    counts = best_init_counts
    current_L = best_init_L
    best_L = current_L
    best_residues = residues.copy()

    # --- Phase 2: Local Search / Simulated Annealing ---
    # Strategy: change one or a few residues at a time, accept if L improves,
    # or with some probability if L doesn't decrease too much.
    
    T = 50.0
    cooling_rate = 0.99995
    min_T = 0.5
    
    # Precompute for faster updates
    # For each prime index, precompute the positions array for each possible residue?
    # That's too much memory. Instead, use the slice approach but optimize.
    
    # Let's precompute for each prime the list of positions for residue 0, 
    # and then for residue r, positions are (pos0 + r) % p mapped back... 
    # Actually it's simpler: for prime p, residue r, positions are r, r+p, r+2p, ...
    # If r=0: p, 2p, ...
    # The difference between residue r1 and r2 for the same prime p:
    # positions(r1) = {r1 + kp : k >= 0, r1 + kp <= MAX_L} (with r1=0 meaning p, 2p, ...)
    # positions(r2) = {r2 + kp : k >= 0, r2 + kp <= MAX_L}
    # The set difference can be computed but it's complex.
    # Let's just do the remove/add with slices.
    
    while time.time() < deadline:
        # Choose how many primes to change
        if random.random() < 0.7:
            K = 1
        elif random.random() < 0.9:
            K = 2
        else:
            K = random.randint(3, 5)
        
        K = min(K, n_primes)
        indices_to_change = random.sample(range(n_primes), K)
        
        old_residues = [residues[i] for i in indices_to_change]
        
        # Generate new random residues
        new_residues = []
        for i in indices_to_change:
            p = primes[i]
            new_residues.append(random.randint(p))
        
        # Apply changes
        for j, idx in enumerate(indices_to_change):
            p = primes[idx]
            old_r = residues[idx]
            new_r = new_residues[j]
            
            if old_r == new_r:
                continue
            
            # Remove old
            start_old = old_r if old_r > 0 else p
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
            
            # Add new
            start_new = new_r if new_r > 0 else p
            if start_new <= MAX_L:
                counts[start_new::p] += 1
            
            residues[idx] = new_r
        
        # Calculate new L
        new_L = find_L(counts)
        
        # Acceptance criterion
        if new_L >= current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            delta = new_L - current_L
            # Accept bad moves with probability exp(delta/T)
            if delta > -1000:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_L = new_L
                else:
                    # Revert
                    for j, idx in enumerate(indices_to_change):
                        p = primes[idx]
                        new_r = new_residues[j]
                        old_r = old_residues[j]
                        
                        start_new = new_r if new_r > 0 else p
                        if start_new <= MAX_L:
                            counts[start_new::p] -= 1
                        
                        start_old = old_r if old_r > 0 else p
                        if start_old <= MAX_L:
                            counts[start_old::p] += 1
                        
                        residues[idx] = old_r
            else:
                # Revert
                for j, idx in enumerate(indices_to_change):
                    p = primes[idx]
                    new_r = new_residues[j]
                    old_r = old_residues[j]
                    
                    start_new = new_r if new_r > 0 else p
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                    
                    start_old = old_r if old_r > 0 else p
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
                    
                    residues[idx] = old_r
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    return best_residues
