"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _compute_score(residues, primes, max_check=100000):
    """
    Compute the largest L such that every integer 1..L satisfies at least two congruences.
    Uses an array to count hits for each number in 1..max_check.
    """
    # Count hits for each number x in 1..max_check
    # We only care about x >= 1
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for r, p in zip(residues, primes):
        # x = r (mod p) => x = r, r+p, r+2p, ...
        # Note: x must be >= 1.
        # If r == 0, the first positive multiple is p (since 0 is not in 1..L usually, 
        # but the problem says integers 1..L. 0 mod p is p, 2p, etc? 
        # Wait, x = r (mod p). If r=0, x is a multiple of p. 
        # The smallest positive multiple is p.
        # If r > 0, the smallest positive is r.
        
        start = r if r > 0 else p
        # Generate indices: start, start+p, start+2p, ... <= max_check
        # Using np.arange for speed
        indices = np.arange(start, max_check + 1, p)
        counts[indices] += 1
        
    # Find the largest L such that counts[1:L+1] are all >= 2
    # We can check from 1 upwards
    # A vectorized way: find the first index where count < 2
    # But we need the prefix property.
    # Let's just iterate or use a cumulative check.
    # Since max_check is 10^5, a simple loop is fast enough in Python if optimized, 
    # but numpy is better.
    
    # Create a boolean mask where count >= 2
    valid = counts >= 2
    # valid[0] is irrelevant, we start at 1.
    # We want the first False in valid[1:]
    # If all are True, L = max_check.
    
    # np.argmax returns the index of the first True. 
    # We want the first False.
    # ~valid gives True where count < 2.
    first_invalid = np.argmax(~valid[1:])
    
    # If the first element of ~valid[1:] is False (i.e., valid[1] is True), 
    # argmax returns 0 if the first element is True? 
    # No, argmax returns the index of the first occurrence of the maximum value.
    # If ~valid[1:] is [False, False, True, ...], max is True (1). 
    # Wait, bools are 0/1. 
    # If ~valid[1:] = [0, 0, 1, ...], argmax returns 2. 
    # This means indices 1 and 2 (in 1-based 1..L logic: 1, 2) are valid?
    # Let's trace carefully.
    # valid[1] is for x=1.
    # ~valid[1] is for x=1.
    # If x=1 is invalid, ~valid[1] is True (1). argmax returns 0. L should be 0.
    # If x=1 is valid, ~valid[1] is False (0).
    # If x=2 is invalid, ~valid[2] is True (1). argmax returns 1. L should be 1.
    # So L = first_invalid.
    
    # Edge case: if all are valid, ~valid[1:] is all False (0). 
    # argmax of all zeros returns 0. 
    # So we need to check if any True exists.
    if not np.any(~valid[1:]):
        return max_check
    else:
        return first_invalid

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # Initial residues: 0 for all
    residues = [0] * n_primes
    
    # We want to maximize L.
    # Strategy: 
    # 1. Start with a small L and try to extend it.
    # 2. At each step, identify the "bottleneck" integers in 1..L where count < 2.
    # 3. For each such integer, find a prime p and a residue r_p such that 
    #    changing r_p to x % p increases the count for x and minimally hurts others.
    # This is complex. 
    
    # Alternative Strategy: Randomized Local Search with Greedy Initialization
    # 1. Initialize residues randomly.
    # 2. Compute L.
    # 3. If L is small, identify the first gap x* where count < 2.
    # 4. Try to fix x* by adjusting residues of primes.
    #    For a prime p, if we set r_p = x* % p, x* gets a hit.
    #    We should pick primes where this change is "cheap" (doesn't break other coverage).
    
    # Let's try a specific greedy construction:
    # We want to cover 1, 2, 3, ... 
    # For each x in 1..TargetL:
    #   We need 2 primes p1, p2 such that r_p1 = x % p1 and r_p2 = x % p2.
    #   We can assign residues to primes as we go.
    #   However, a prime p covers x, x+p, x+2p...
    #   So assigning r_p affects many x's.
    
    # Let's use a "Greedy Set Cover" like approach for the prefix.
    # We maintain counts for 1..MaxL.
    # We want to extend the prefix L.
    # Let current L be the max prefix with count >= 2.
    # To extend L, we need to fix x = L+1.
    # We need to increase count[x] by at least 2 (if it's 0) or 1 (if it's 1).
    # We can change residues of up to 2 primes (or more) to hit x.
    # Which primes to change? 
    # Changing r_p to x % p hits x. 
    # It also hits x-p, x+p, etc.
    # We want to avoid decreasing counts of y <= L.
    
    # Let's implement a simple iterative improvement:
    # 1. Start with random residues.
    # 2. Calculate L.
    # 3. While time permits:
    #    a. Find the first gap x_gap.
    #    b. If no gap, try to push L higher (maybe by checking further).
    #    c. If gap exists, try to fix it.
    #       - Identify primes p where current r_p != x_gap % p.
    #       - For a random subset of these primes, try changing r_p to x_gap % p.
    #       - Check if the new L is larger. If yes, keep it.
    #       - If no, revert.
    #    d. To prevent getting stuck, occasionally do a random perturbation.
    
    np.random.seed(42)
    
    # Initialize with a better heuristic: 
    # Assign r_p = (p+1)//2 or similar? 
    # Or just random.
    residues = np.array([np.random.randint(0, p) for p in primes])
    
    best_residues = residues.copy()
    best_L = _compute_score(best_residues, primes, 100000)
    
    # Greedy improvement loop
    # We focus on the first gap.
    # To speed up _compute_score, we can update counts incrementally?
    # For now, full recompute is ~16.8M ops. In numpy, this is very fast (<0.1s).
    # 100s / 0.05s = 2000 iterations.
    
    current_residues = best_residues.copy()
    current_L = best_L
    
    # We need to know the counts to find the gap efficiently.
    # Let's write a helper to get counts and L.
    def get_counts_and_L(res, primes, max_check=100000):
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(res, primes):
            start = r if r > 0 else p
            if start <= max_check:
                indices = np.arange(start, max_check + 1, p)
                counts[indices] += 1
        valid = counts >= 2
        if not np.any(~valid[1:]):
            return counts, max_check
        first_invalid = np.argmax(~valid[1:])
        return counts, first_invalid

    counts, current_L = get_counts_and_L(current_residues, primes)
    
    while time.time() < deadline:
        if current_L == 100000:
            break
            
        # Find the first gap
        # We have counts. 
        # gap_idx is the first x in 1..100000 where counts[x] < 2.
        # This is current_L + 1 if current_L < 100000.
        # Wait, _compute_score returns the largest L such that 1..L are valid.
        # So the gap is at current_L + 1.
        
        x_gap = current_L + 1
        
        if x_gap > 100000:
            break
            
        # We need to increase counts[x_gap] to >= 2.
        # Current count is c = counts[x_gap].
        # We need to add (2 - c) hits.
        # We can change residues of primes.
        
        # Identify candidates: primes p where current residue r_p != x_gap % p.
        # Changing r_p to x_gap % p will add 1 to counts[x_gap].
        # It will remove 1 from counts[x_gap - k*p] for k>=1 (if positive) 
        # and add 1 to counts[x_gap + k*p] for k>=1.
        # Actually, it removes the old hit and adds the new hit.
        # Old hit: y_old = x_gap - (x_gap % p) ... no.
        # Old residue r_old. Hits: r_old, r_old+p, ...
        # New residue r_new = x_gap % p. Hits: r_new, r_new+p, ...
        # x_gap is a hit for new residue.
        # Is x_gap a hit for old residue? 
        # x_gap % p == r_old? 
        # If yes, changing does nothing for x_gap.
        # We only consider p where x_gap % p != r_p.
        
        # To avoid breaking the prefix 1..current_L:
        # We must ensure that for any y in 1..current_L, 
        # if y was hit by the old residue, and is NOT hit by the new residue,
        # then counts[y] must remain >= 2.
        # i.e., counts[y] - 1 >= 2 => counts[y] >= 3.
        
        # Let's pick a random prime p to try changing.
        # Or pick the "best" prime.
        
        # Heuristic: Pick a prime p such that the "cost" of changing is low.
        # Cost = number of y in 1..current_L where counts[y] == 2 and 
        #        (y is hit by old residue) XOR (y is hit by new residue).
        # Actually, if y is hit by old and not new, count drops by 1. 
        # If count was 2, it becomes 1 -> Breaks prefix.
        # So we need counts[y] >= 3 for all such y.
        
        # Let's just try a few random primes.
        
        # Get list of indices of primes
        # We can just iterate through a random sample of prime indices.
        
        improved = False
        # Try 10 random primes
        for _ in range(10):
            if time.time() > deadline:
                break
                
            p_idx = np.random.randint(0, len(primes))
            p = primes[p_idx]
            
            r_old = current_residues[p_idx]
            r_new = x_gap % p
            
            if r_old == r_new:
                continue
                
            # Check if this change is safe for the prefix 1..current_L
            # We need to check y in 1..current_L.
            # y is affected if y % p == r_old or y % p == r_new.
            # Specifically:
            # - If y % p == r_old and y % p != r_new: count[y] decreases by 1.
            # - If y % p == r_new and y % p != r_old: count[y] increases by 1.
            # - If y % p == r_old == r_new: no change (handled by r_old != r_new check).
            
            # We only care about decreases.
            # So we need to check y in 1..current_L such that y % p == r_old.
            # For these y, we need counts[y] >= 3.
            
            # Find such y's.
            # y = r_old + k*p.
            # If r_old == 0, y = p, 2p, ...
            # If r_old > 0, y = r_old, r_old+p, ...
            
            start_y = r_old if r_old > 0 else p
            # Check if start_y <= current_L
            if start_y <= current_L:
                # Iterate y
                safe = True
                y = start_y
                while y <= current_L:
                    if counts[y] < 3:
                        safe = False
                        break
                    y += p
                if not safe:
                    continue
            
            # If safe, apply the change
            current_residues[p_idx] = r_new
            # Update counts locally? 
            # Or just recompute? 
            # Recomputing is ~0.05s. 
            # Let's recompute to be safe and simple.
            
            new_counts, new_L = get_counts_and_L(current_residues, primes)
            
            if new_L > current_L:
                current_L = new_L
                counts = new_counts
                best_L = current_L
                best_residues = current_residues.copy()
                improved = True
                break # Found an improvement, restart gap search
            else:
                # Revert
                current_residues[p_idx] = r_old
                
        if not improved:
            # Stuck. Try a random perturbation of a few residues to escape local minimum.
            # Pick 2-3 random primes and change their residues randomly.
            for _ in range(3):
                p_idx = np.random.randint(0, len(primes))
                old_r = current_residues[p_idx]
                current_residues[p_idx] = np.random.randint(0, primes[p_idx])
            
            new_counts, new_L = get_counts_and_L(current_residues, primes)
            if new_L > current_L:
                current_L = new_L
                counts = new_counts
                best_L = current_L
                best_residues = current_residues.copy()
            else:
                # Revert perturbation if it didn't help immediately? 
                # Or keep it to explore? 
                # Let's revert to best known to maintain stability, 
                # but we are stuck, so maybe keep the perturbation to try more steps?
                # If we revert, we are in the same state. 
                # Let's keep the perturbation and continue. 
                # But we need to update counts.
                counts = new_counts
                # Note: current_L might have decreased. 
                # We should track the global best, but use current state for search.
                # If current_L < best_L, we are in a worse state.
                # This is a random walk. 
                # To avoid drifting too far, we can revert if it gets much worse?
                # For now, let's just continue. The next iteration will try to fix the new gap.

    return list(best_residues)

# EVOLVE-BLOCK-END
