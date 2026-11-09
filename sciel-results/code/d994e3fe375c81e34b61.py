"""CSP-based search for prime residue covering system."""
import time
from fractions import Fraction
import numpy as np
import random

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _covered_length(classes, max_L=100000):
    """Compute the largest L such that all integers 1..L are covered using NumPy."""
    if not classes:
        return 0
    
    # Use a boolean array for coverage marking. 
    # Index 0 is unused (we care about 1..max_L).
    covered = np.zeros(max_L + 1, dtype=bool)
    
    for p, r in classes:
        # We want to mark n such that n % p == r and 1 <= n <= max_L
        # The first number >= 1 satisfying n % p == r is:
        # If r == 0, the first is p.
        # If r != 0, the first is r (if r>=1) or r+p (if r<1, but r>=0 so r=0 is handled).
        # Actually, if r=0, numbers are p, 2p, ...
        # If r>0, numbers are r, r+p, ...
        
        start = r
        if start == 0:
            start = p
        elif start < 1:
            # r is 0..p-1, so r>=1 is always true if r!=0.
            # But just in case, adjust to be >= 1
            start += p
            
        # Ensure start is within bounds
        if start > max_L:
            continue
            
        # Mark the arithmetic progression
        # np.arange(start, max_L+1, p)
        # covered[start::p] = True is faster
        
        covered[start::p] = True
        
    # Find the first gap in 1..max_L
    # We need the first index i >= 1 where covered[i] is False.
    # If all are True, return max_L.
    
    # Efficient way: check if any False in covered[1:]
    # If all True, return max_L
    if not np.any(~covered[1:]):
        return max_L
        
    # Find the first False
    # np.argmax returns the index of the first True. 
    # We want the first False in ~covered.
    first_false_idx = np.argmax(~covered)
    # ~covered[0] is True (since covered[0] is False), so argmax might be 0.
    # We need the first False in covered[1:].
    # Let's just use a loop or a smarter numpy trick.
    # Actually, np.where(~covered)[0] gives all indices.
    # But creating a list of 100k indices is okay if done once, but we do this many times.
    
    # Better: 
    # If covered[1] is False, return 0.
    # We can use:
    # gaps = np.where(~covered[1:])[0]
    # if len(gaps) > 0: return gaps[0]
    # else: return max_L
    
    # Even better:
    # Check if the first element is False
    if not covered[1]:
        return 0
        
    # Use searchsorted or just argmax on the inverted slice?
    # np.argmax(~covered[1:]) gives index in the slice.
    # If the slice is all True, argmax returns 0 (which is wrong, it should be "none").
    # So we check if there is any False first.
    
    # We already checked `if not np.any(~covered[1:])`. So there IS a False.
    # So argmax will find the first one.
    idx_in_slice = np.argmax(~covered[1:])
    # The actual index in 1..max_L is idx_in_slice + 1
    # The length covered is (idx_in_slice + 1) - 1 = idx_in_slice
    return int(idx_in_slice)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}
    
    # Helper to calculate budget sum quickly (using float for speed, Fraction for final check)
    # Since we only care about <= 1, and denominators are small, float is safe for comparison
    # if we add a small epsilon or use exact only at the end.
    # However, the problem requires exact rationals for the constraint.
    # We will use Fraction for the constraint check in the main loop, but it might be slow.
    # Optimization: Precompute float reciprocals for sorting/greedy selection, 
    # and use Fraction only to verify the final solution or when adding/removing.
    
    float_recip = {p: 1.0 / p for p in primes}
    
    best_classes = []
    best_L = 0
    best_budget = Fraction(1)
    
    # Greedy Construction
    # We want to maximize L.
    # Strategy: 
    # 1. Start with empty set.
    # 2. At each step, try adding a prime p (not in set) such that budget allows.
    # 3. For each candidate p, try all residues r.
    # 4. Select the (p, r) that maximizes the new L.
    # 5. Repeat until no improvement or budget exhausted.
    
    # To speed up, we can limit the search space.
    # Small primes are usually more effective for covering short ranges? 
    # Actually, small primes cover denser sets.
    # But large primes cover sparser sets.
    # For covering 1..L, small primes are generally better because they have more residues relative to their density?
    # Let's just try all available primes.
    
    used_primes = set()
    
    # Initial greedy pass
    while time.time() < deadline - 50:
        best_gain = 0
        best_p = None
        best_r = None
        
        # Sort primes by size? Smaller primes might give larger gains initially.
        # Let's iterate over all available primes.
        for p in primes:
            if p in used_primes:
                continue
            if best_budget < recip[p]:
                continue
                
            # Try all residues for this prime
            # Optimization: We don't need to try all if we can estimate.
            # But let's try all for accuracy.
            for r in range(p):
                trial_classes = best_classes + [[p, r]]
                L = _covered_length(trial_classes)
                gain = L - best_L
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            
            # Early break if we found a very large gain?
            # Unlikely to jump by 1000s in one step for large L.
            
        if best_p is not None and best_gain > 0:
            best_classes.append([best_p, best_r])
            used_primes.add(best_p)
            best_L += best_gain
            best_budget -= recip[best_p]
        else:
            break
    
    # Local Search / Refinement
    # Try to improve by:
    # 1. Changing residues of existing primes.
    # 2. Swapping a prime for another.
    
    random.seed(42)
    
    # Limit the number of iterations for local search
    max_iters = 500
    for _ in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        improved = False
        
        # 1. Try changing a random residue
        if best_classes:
            idx = random.randrange(len(best_classes))
            p, r = best_classes[idx]
            old_r = r
            best_r = r
            best_L_trial = best_L
            
            # Try a few random residues or all?
            # All is expensive. Let's try all for the chosen prime.
            for new_r in range(p):
                if new_r == old_r:
                    continue
                best_classes[idx][1] = new_r
                L = _covered_length(best_classes)
                if L > best_L_trial:
                    best_L_trial = L
                    best_r = new_r
            
            if best_r != old_r:
                best_classes[idx][1] = best_r
                best_L = best_L_trial
                improved = True
            else:
                best_classes[idx][1] = old_r
        
        # 2. Try swapping a prime
        if not improved and best_classes:
            idx = random.randrange(len(best_classes))
            p_old, r_old = best_classes[idx]
            
            # Try replacing p_old with a new prime p_new
            # We need to find p_new such that budget allows:
            # current_budget - 1/p_old + 1/p_new <= 1
            # i.e., 1/p_new <= 1/p_old + current_budget
            # current_budget = best_budget
            
            # Iterate over candidates
            best_gain_swap = 0
            best_new_p = None
            best_new_r = None
            
            for p_new in primes:
                if p_new in used_primes and p_new != p_old:
                    continue
                if best_budget - recip[p_old] + recip[p_new] > Fraction(1):
                    continue
                
                # Try residues for p_new
                for r_new in range(p_new):
                    # Create trial classes
                    # Replace p_old with p_new
                    trial_classes = best_classes[:]
                    trial_classes[idx] = [p_new, r_new]
                    
                    L = _covered_length(trial_classes)
                    gain = L - best_L
                    if gain > best_gain_swap:
                        best_gain_swap = gain
                        best_new_p = p_new
                        best_new_r = r_new
            
            if best_new_p is not None and best_gain_swap > 0:
                # Perform swap
                best_classes[idx] = [best_new_p, best_new_r]
                used_primes.remove(p_old)
                used_primes.add(best_new_p)
                best_budget = best_budget - recip[p_old] + recip[best_new_p]
                best_L += best_gain_swap
                improved = True

        # 3. Try adding a new prime if budget allows
        if not improved:
            best_gain_add = 0
            best_new_p = None
            best_new_r = None
            
            for p_new in primes:
                if p_new in used_primes:
                    continue
                if best_budget < recip[p_new]:
                    continue
                
                for r_new in range(p_new):
                    trial_classes = best_classes + [[p_new, r_new]]
                    L = _covered_length(trial_classes)
                    gain = L - best_L
                    if gain > best_gain_add:
                        best_gain_add = gain
                        best_new_p = p_new
                        best_new_r = r_new
            
            if best_new_p is not None and best_gain_add > 0:
                best_classes.append([best_new_p, best_new_r])
                used_primes.add(best_new_p)
                best_budget -= recip[best_new_p]
                best_L += best_gain_add
                improved = True

        if not improved:
            # Try a random restart of residues for all classes?
            # Or just break if we've been stuck for a while?
            # Let's try a "kick": randomize residues for a few classes.
            if random.random() < 0.1:
                old_classes = [c[:] for c in best_classes]
                for _ in range(3):
                    if best_classes:
                        idx = random.randrange(len(best_classes))
                        p = best_classes[idx][0]
                        best_classes[idx][1] = random.randrange(p)
                
                L = _covered_length(best_classes)
                if L > best_L:
                    best_L = L
                else:
                    best_classes = old_classes

    return best_classes
