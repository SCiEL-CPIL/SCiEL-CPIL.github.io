import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Greedy Initialization
    # We want to ensure coverage >= 2 for x in [1, L_init]
    L_init = 200
    residues = [0] * n_primes
    # Initialize coverage array
    # We will simulate the greedy fill
    # To do this efficiently, we can just pick residues randomly first, 
    # then iterate x from 1 to L_init and fix gaps.
    
    random.seed(42)
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Function to compute coverage array
    def compute_cov(res, L):
        cov = np.zeros(L + 1, dtype=np.int32)
        for r, p in zip(res, primes):
            cov[r::p] += 1
        return cov
    
    # Greedy repair for initialization
    for x in range(1, L_init + 1):
        # Check current coverage of x
        count = 0
        for i, p in enumerate(primes):
            if x % p == residues[i]:
                count += 1
        if count < 2:
            # Need to add coverage
            # Pick a prime that doesn't cover x
            candidates = [i for i, p in enumerate(primes) if x % p != residues[i]]
            if candidates:
                idx = random.choice(candidates)
                residues[idx] = x % primes[idx]
    
    # Now we have a decent start.
    # Local Search
    current_residues = residues[:]
    best_residues = residues[:]
    best_score = 0
    
    # Start with a small check limit
    L_check = 500
    cov = compute_cov(current_residues, L_check)
    
    # Find initial score
    # Find first index where cov < 2
    # cov[0] is not checked, we care about 1..L
    # Note: cov is 0-indexed. x=1 is cov[1].
    
    def find_first_gap(cov_arr, limit):
        # Find first x in [1, limit] where cov_arr[x] < 2
        # Using argmax on boolean array
        # If no gap, return limit + 1
        gaps = np.where(cov_arr[1:limit+1] < 2)[0]
        if len(gaps) == 0:
            return limit + 1
        return gaps[0] + 1 # +1 because we sliced from 1
        
    current_gap = find_first_gap(cov, L_check)
    current_score = current_gap - 1
    if current_score > best_score:
        best_score = current_score
        best_residues = current_residues[:]
        
    stagnant_count = 0
    stagnation_threshold = 100
    
    while time.time() < deadline:
        # Check if we need to expand the window
        if current_gap > L_check:
            # We have covered up to L_check
            # Expand L_check
            L_check = int(L_check * 1.5) + 100
            if L_check > 100000:
                L_check = 100000
            cov = compute_cov(current_residues, L_check)
            current_gap = find_first_gap(cov, L_check)
            current_score = current_gap - 1
            if current_score > best_score:
                best_score = current_score
                best_residues = current_residues[:]
                stagnant_count = 0
            continue
        
        # We have a gap at current_gap
        # Try to fix it
        x_gap = current_gap
        # Find primes that do NOT cover x_gap
        # r_p != x_gap % p
        candidates = []
        for i, p in enumerate(primes):
            if current_residues[i] != x_gap % p:
                candidates.append(i)
                
        if not candidates:
            # All primes cover x_gap? Then coverage is 168 >= 2.
            # This shouldn't happen if current_gap is a gap.
            # Unless my find_first_gap is wrong or cov is stale.
            # Recompute cov to be sure
            cov = compute_cov(current_residues, L_check)
            current_gap = find_first_gap(cov, L_check)
            continue
            
        # Pick a random candidate
        # To be smarter, we could pick a prime with large p (affects fewer spots)
        # or small p (affects many spots, riskier but might fix more?)
        # Let's stick to random for now, or weighted by 1/p?
        # Large primes are "safer" to change because they cover fewer numbers in [1, L].
        # Let's bias towards larger primes.
        # Weights: 1/p is small for large p. We want large p. So weight p?
        # Let's just pick random from candidates.
        
        improved = False
        # Try a few random candidates if the first one fails?
        # No, just one, then revert if it fails.
        
        idx = random.choice(candidates)
        old_r = current_residues[idx]
        new_r = x_gap % primes[idx]
        
        # Apply change
        current_residues[idx] = new_r
        
        # Update cov incrementally
        p = primes[idx]
        # Subtract old
        cov[old_r::p] -= 1
        # Add new
        cov[new_r::p] += 1
        
        # Find new gap
        new_gap = find_first_gap(cov, L_check)
        new_score = new_gap - 1
        
        if new_score > current_score:
            # Accept
            current_score = new_score
            current_gap = new_gap
            stagnant_count = 0
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues[:]
        else:
            # Revert
            current_residues[idx] = old_r
            # Revert cov
            cov[new_r::p] -= 1
            cov[old_r::p] += 1
            stagnant_count += 1
            
            # If stagnant, try a random mutation
            if stagnant_count > stagnation_threshold:
                stagnant_count = 0
                # Random mutation
                idx = random.randint(0, n_primes - 1)
                old_r = current_residues[idx]
                new_r = random.randint(0, primes[idx] - 1)
                current_residues[idx] = new_r
                p = primes[idx]
                cov[old_r::p] -= 1
                cov[new_r::p] += 1
                
                new_gap = find_first_gap(cov, L_check)
                new_score = new_gap - 1
                
                if new_score > current_score:
                    current_score = new_score
                    current_gap = new_gap
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = current_residues[:]
                else:
                    # Revert
                    current_residues[idx] = old_r
                    cov[new_r::p] -= 1
                    cov[old_r::p] += 1
    
    # Final evaluation with larger limit to ensure we capture the true score
    # The search might have stopped with a small L_check, but the solution might be better.
    # However, the search dynamically increased L_check.
    # Just to be safe, compute score for best_residues with a large limit.
    # But we don't have the cov array for best_residues at large L.
    # We can just return best_residues. The evaluator will compute the score.
    
    return best_residues
