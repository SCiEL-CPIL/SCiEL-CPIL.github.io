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
    deadline = start_time + 110  # 10s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Random Initialization
    random.seed(42)
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Set a reasonable L_check. 
    # The task cares about L >= 1000. 
    # Using 5000 allows us to verify if we've beaten 1000 significantly.
    L_check = 5000
    
    # Compute initial coverage
    cov = np.zeros(L_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        cov[r::p] += 1
        
    # Function to find first gap in cov[1:limit+1]
    # Returns the first x in [1, limit] such that cov[x] < 2, or limit+1 if none.
    def find_first_gap(cov_arr, start_idx, limit):
        # We want the first index i >= start_idx such that cov_arr[i] < 2
        # Slice from start_idx to limit+1
        segment = cov_arr[start_idx:limit+1]
        if segment.size == 0:
            return limit + 1
        # Find first True in (segment < 2)
        # argmax returns 0 if all are False (no gap), which corresponds to start_idx
        # But if there is a gap, it returns the offset.
        # If no gap, argmax returns 0. We need to distinguish "no gap" from "gap at start_idx".
        # If cov[start_idx] < 2, gap is at start_idx.
        # Let's just use np.where for simplicity if the slice is small, or a loop?
        # np.where is fast for arrays.
        gaps = np.where(segment < 2)[0]
        if len(gaps) == 0:
            return limit + 1
        return start_idx + gaps[0]
        
    current_score = find_first_gap(cov, 1, L_check) - 1
    best_residues = residues[:]
    best_score = current_score
    
    stagnant_count = 0
    stagnation_threshold = 500
    
    while time.time() < deadline:
        if current_score >= L_check:
            # We have covered up to L_check. 
            # We can try to expand, but for now, let's just return if we are doing well.
            # Or we can increase L_check slightly and re-evaluate?
            # Re-evaluating is O(N*L). 
            # Let's just return best_residues if we've been stable.
            # Actually, if current_score is high, we are good.
            # Let's try a few more iterations to see if we can break the barrier.
            # But if we are at 5000, we are way above 1000.
            break
            
        x_gap = current_score + 1
        
        # Find a prime that does NOT currently cover x_gap
        # We want to change a prime's residue to x_gap % p
        # If it already covers x_gap, changing it might remove coverage from other spots, 
        # but it won't help cover x_gap (since it already does). 
        # So we must pick a prime where residues[i] != x_gap % primes[i].
        
        # To be fast, we just pick a random index and check.
        # If it's invalid, pick another.
        # Or pick 5 random candidates and try the first valid one.
        
        improved = False
        # Try up to 10 random candidates to find a valid move
        for _ in range(10):
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            target_r = x_gap % p
            old_r = residues[idx]
            
            if old_r == target_r:
                continue # This prime already covers x_gap, changing it is useless for covering x_gap
            
            # Try the move
            residues[idx] = target_r
            cov[old_r::p] -= 1
            cov[target_r::p] += 1
            
            # Find new gap
            # The gap was at x_gap. It might be fixed.
            # New gap is >= x_gap.
            new_score = find_first_gap(cov, x_gap, L_check) - 1
            
            if new_score > current_score:
                # Accept
                current_score = new_score
                stagnant_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
                improved = True
                break
            else:
                # Revert
                residues[idx] = old_r
                cov[target_r::p] -= 1
                cov[old_r::p] += 1
        
        if not improved:
            stagnant_count += 1
            if stagnant_count > stagnation_threshold:
                # Random restart or mutation?
                # A simple random mutation of one residue
                idx = random.randint(0, n_primes - 1)
                old_r = residues[idx]
                new_r = random.randint(0, primes[idx] - 1)
                residues[idx] = new_r
                p = primes[idx]
                cov[old_r::p] -= 1
                cov[new_r::p] += 1
                
                new_score = find_first_gap(cov, 1, L_check) - 1
                if new_score > current_score:
                    current_score = new_score
                    stagnant_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues[:]
                else:
                    # Revert
                    residues[idx] = old_r
                    cov[new_r::p] -= 1
                    cov[old_r::p] += 1
                    stagnant_count = 0 # Reset to try different mutations
    
    return best_residues
