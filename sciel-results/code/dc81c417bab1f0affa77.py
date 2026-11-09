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

def _find_first_gap(residues, primes, max_check=2000):
    """Find the smallest x >= 1 such that coverage < 2. Returns x, or max_check+1 if none."""
    for x in range(1, max_check + 1):
        count = 0
        for i, p in enumerate(primes):
            if x % p == residues[i]:
                count += 1
                if count >= 2:
                    break
        if count < 2:
            return x
    return max_check + 1

def _score(residues, primes, limit=2000):
    """Returns the length L. Uses _find_first_gap."""
    gap = _find_first_gap(residues, primes, limit)
    return gap - 1

def construct():
    start_time = time.time()
    deadline = start_time + 115 # 5s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Better heuristic initialization:
    # For each prime p, set r_p to cover small numbers densely.
    # A simple strategy: for p=2, r=0 (covers evens). For p=3, r=1 (covers 1,4,7...).
    # This doesn't guarantee double coverage, but it's a start.
    # Let's try a smarter init: for each prime p, pick r_p such that it covers many small numbers.
    # Actually, just random init is fine, but let's try a few different seeds and pick the best.
    
    best_residues = None
    best_score = -1
    
    for seed in [42, 123, 999, 2024]:
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        score = _score(residues, primes, 2000)
        if score > best_score:
            best_score = score
            best_residues = residues
    
    # Local Search with Repair
    current_residues = best_residues[:]
    current_score = best_score
    
    max_iter = 10000
    stagnant_count = 0
    stagnation_threshold = 500
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Find first gap
        gap = _find_first_gap(current_residues, primes, 2000)
        
        if gap > 2000:
            # We have at least 2000 coverage.
            # Try to extend the check range or just stop improving this metric?
            # The score function returns gap-1.
            # If gap is 2001, score is 2000.
            # We can try to push it higher by checking further?
            # For now, if we hit the cap, we can try random moves to see if we can "stumble" past it.
            # Or just break if we've been stuck?
            # Let's try a few random moves if we hit the cap.
            if i % 100 == 0:
                idx = random.randint(0, n_primes - 1)
                old_r = current_residues[idx]
                new_r = random.randint(0, primes[idx] - 1)
                current_residues[idx] = new_r
                new_score = _score(current_residues, primes, 2000)
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
                else:
                    current_residues[idx] = old_r
                    stagnant_count += 1
            continue
            
        # gap is <= 2000. We need to fix it.
        # Identify primes that CAN cover `gap`.
        # These are primes p where current_r_p != gap % p.
        # If we change r_p to gap % p, gap gets +1 coverage.
        
        candidates = []
        for idx, p in enumerate(primes):
            target_r = gap % p
            if current_residues[idx] != target_r:
                candidates.append((idx, target_r))
        
        if not candidates:
            # All primes already have r_p == gap % p, but coverage < 2?
            # This means fewer than 2 primes exist? Or logic error.
            # With 168 primes, this shouldn't happen unless gap is covered by < 2 primes 
            # and all of them are already set? 
            # Wait, if all primes are set to gap%p, then ALL 168 cover gap.
            # So coverage is 168 >= 2.
            # So this case implies my `_find_first_gap` or logic is wrong.
            # Or `gap` was identified as < 2, but if all are set, it's 168.
            # So `candidates` will not be empty if gap is truly a gap.
            # If gap is a gap, then count of p with r_p == gap%p is < 2.
            # So for most p, r_p != gap%p.
            # So candidates should be large.
            break
            
        # Try to fix the gap.
        # We can't try all candidates (up to 168) * all possible moves?
        # We only need to change ONE residue to gap%p to increase coverage by 1.
        # But we might break others.
        
        # Strategy: Pick a random candidate, or the one that seems "safest".
        # "Safest" is hard to determine quickly.
        # Let's try a few random candidates.
        
        improved = False
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        # Try more candidates if we are not making progress
        num_candidates_to_try = 20 if stagnant_count > 100 else 10
        
        for idx, target_r in candidates[:num_candidates_to_try]: # Try up to N random candidates
            old_r = current_residues[idx]
            current_residues[idx] = target_r
            new_score = _score(current_residues, primes, 2000)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
                break
            else:
                current_residues[idx] = old_r
        
        if not improved:
            # No single fix worked. Try a random perturbation to escape local minimum.
            idx = random.randint(0, n_primes - 1)
            old_r = current_residues[idx]
            new_r = random.randint(0, primes[idx] - 1)
            current_residues[idx] = new_r
            new_score = _score(current_residues, primes, 2000)
            if new_score >= current_score: # Accept neutral or better
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    stagnant_count = 0
            else:
                # Revert? Or keep to explore? 
                # Let's revert to keep progress steady, but occasionally accept bad moves.
                if random.random() < 0.05:
                    current_score = new_score
                    stagnant_count += 1
                else:
                    current_residues[idx] = old_r
                    stagnant_count += 1
            
            # If we've been stagnant for too long, do a bigger restart
            if stagnant_count > stagnation_threshold:
                stagnant_count = 0
                # Restart from best found so far
                current_residues = best_residues[:]
                current_score = best_score
                # Or do a random restart? Let's stick with best for now.

    # Final evaluation with a higher limit to get the true score
    # We can run _find_first_gap with a larger max_check on best_residues
    final_gap = _find_first_gap(best_residues, primes, 100000)
    # If it passed 100000, score is 100000.
    # The evaluator caps at 10^5.
    
    return best_residues
