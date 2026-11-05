import numpy as np
import time

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
    deadline = start_time + 140.0 # 10s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000
    
    # Greedy Initialization
    # We focus on the first M integers for the greedy choice to keep it fast
    M = 5000 
    residues = [0] * n_primes
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    for i, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        
        # We want to maximize the number of x in [1, M] where counts[x] becomes >= 2
        # Specifically, we look for x where counts[x] == 1 and x % p == r
        # Or x where counts[x] == 0 and x % p == r (less important)
        
        for r in range(p):
            gain = 0
            start = r if r > 0 else p
            if start > M:
                continue
            # Iterate through positions start, start+p, ... <= M
            for pos in range(start, M + 1, p):
                if counts[pos] == 1:
                    gain += 2 # Higher weight for making 1->2
                elif counts[pos] == 0:
                    gain += 1 # Lower weight for making 0->1
            
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[i] = best_r
        # Update counts for the full MAX_L range
        start = best_r if best_r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1

    # Calculate initial L
    def calc_L(counts, current_L):
        if current_L > 0:
            # Check if current prefix is still valid
            prefix = counts[1:current_L+1]
            if np.min(prefix) < 2:
                bad = np.where(prefix < 2)[0]
                return int(bad[0])
            # Else, try to extend
            L = current_L
            rest = counts[L+1:]
            if len(rest) == 0:
                return MAX_L
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            return L + int(bad_idx[0])
        else:
            bad_idx = np.where(counts[1:] < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            return int(bad_idx[0])

    current_L = calc_L(counts, 0)
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    
    # Random state for reproducibility in this run
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Pick random prime
        i = rng.integers(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = rng.integers(p)
        
        if new_r == old_r:
            continue
            
        # Apply change
        start_old = old_r if old_r > 0 else p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
            
        start_new = new_r if new_r > 0 else p
        if start_new <= MAX_L:
            counts[start_new::p] += 1
            
        # Calculate new L
        new_L = calc_L(counts, current_L)
        
        # Acceptance
        if new_L >= current_L:
            # Accept
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Reject or Accept with probability
            delta = new_L - current_L
            prob = np.exp(delta / T)
            if rng.random() < prob:
                residues[i] = new_r
                current_L = new_L
            else:
                # Revert
                residues[i] = old_r
                # Revert counts
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                # current_L remains unchanged
                
        T *= cooling
        if T < 1.0:
            T = 1.0
            
    return best_residues
