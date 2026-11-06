"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin before 150s kill
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We need to check a range large enough to potentially exceed 1000.
    # The theoretical bound is loose, but 10000 is a good practical upper bound for search.
    # If we reach 10000, we return 10000 (capped by evaluator logic implicitly, but we return the score).
    max_check = 10000
    
    # Initialize residues to 0 (arbitrary start)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # counts[x] = number of primes p such that x == residues[idx_of_p] (mod p)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    def update_counts(idx, old_r, new_r, delta):
        """
        Update counts array.
        delta = +1 means adding the residue, -1 means removing.
        """
        p = primes[idx]
        if delta == 1:
            r = new_r
            start = r if r > 0 else p
            if start <= max_check:
                counts[start::p] += 1
        else: # delta == -1
            r = old_r
            start = r if r > 0 else p
            if start <= max_check:
                counts[start::p] -= 1

    # Initial setup
    for i in range(n_primes):
        update_counts(i, 0, residues[i], 1)
        
    def get_score():
        """
        Find the largest L such that counts[1..L] >= 2.
        """
        # We want the first index in 1..max_check where count < 2
        # counts[0] is not used for the score (score is for 1..L)
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]

    best_score = get_score()
    best_residues = residues.copy()
    
    # Greedy Strategy:
    # While time remains:
    # 1. Find the current bottleneck B (smallest x in 1..L+1 with count < 2).
    #    Actually, if score is L, then L+1 is the first failure.
    #    So we look at x = best_score + 1.
    # 2. Identify which primes p can cover x. i.e., x % p == r_p for some r_p in [0, p-1].
    #    Actually, ANY prime p can cover x by setting r_p = x % p.
    #    However, changing r_p affects other numbers.
    # 3. We want to pick a prime p and a new residue r such that:
    #    - x becomes covered (count[x] increases by 1).
    #    - We don't break coverage for numbers < x (they must remain >= 2).
    #    - Ideally, we help cover numbers > x as well.
    
    # To make this efficient, we don't try ALL primes. We prioritize:
    # 1. Primes that are small (affect more numbers in the range).
    # 2. Primes whose current residue is "far" from x % p (less disruption? No, we need to calculate exact impact).
    
    # A simpler, robust greedy:
    # Focus on the first failure point x = L + 1.
    # We need count[x] to become >= 2.
    # Currently count[x] is 0 or 1.
    # If count[x] == 1, we need +1. If count[x] == 0, we need +2.
    # Changing one residue can add at most 1 to count[x].
    # So if count[x] == 0, we need to change TWO residues.
    # If count[x] == 1, we need to change ONE residue.
    
    # Let's implement a loop that tries to fix the first failure.
    
    max_iterations = 50000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iterations:
        iter_count += 1
        
        current_score = get_score()
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        if current_score >= max_check:
            break
            
        x_target = current_score + 1
        current_count_x = counts[x_target]
        
        # We need to increase count[x_target] by (2 - current_count_x)
        # If current_count_x >= 2, this shouldn't happen as x_target is the first failure.
        
        # Candidate primes:
        # We should prefer primes that are currently NOT covering x_target, 
        # so that changing them adds coverage without removing it.
        # Also, we should prefer primes where the change doesn't break existing double coverage elsewhere.
        
        # To limit search space, we only consider the first K primes or a random subset?
        # Small primes have larger "footprint" in the range [1, max_check].
        # Changing r for p=2 affects 5000 numbers. Changing r for p=997 affects 10 numbers.
        # So small primes are high-impact.
        
        # Strategy:
        # Try changing residues for primes in increasing order (smallest p first).
        # For each prime p, calculate the "cost" of changing its residue to x_target % p.
        # Cost = number of positions y in [1, x_target-1] where count[y] == 2 and y is covered by p with old_r but NOT with new_r.
        # If Cost == 0, it's a "free" win.
        # If no free win, pick the one with minimum cost? Or just take the first one that doesn't break anything?
        # Actually, if we break something < x_target, the score drops. We must avoid that.
        
        # So, constraint: For all y in [1, x_target-1], if count[y] == 2, we must ensure that after change, count[y] >= 2.
        # This means we cannot remove the ONLY second coverage for any y < x_target.
        
        # Let's find the "critical" positions y < x_target where count[y] == 2.
        # For a candidate change (p, old_r -> new_r):
        # The positions affected are those y where y == old_r (mod p) AND y != new_r (mod p).
        # Among these, if any y < x_target has count[y] == 2, the change is INVALID (breaks prefix).
        
        # Optimization:
        # Precompute for each prime p, the set of positions it covers? No, too memory heavy.
        # Just check the affected positions.
        
        # For a given p and new_r = x_target % p:
        # Affected positions are y = k*p + old_r (for y <= max_check) such that y != k'*p + new_r.
        # Actually, simpler:
        # Old coverage: y % p == old_r
        # New coverage: y % p == new_r
        # Positions losing coverage: y where y % p == old_r AND y % p != new_r.
        # Since old_r != new_r (otherwise no change), ALL positions covered by old_r lose coverage.
        # Wait, if old_r == new_r, no change. We assume old_r != new_r.
        # So, any y covered by p with old_r loses 1 count.
        # We must ensure that for all y < x_target covered by p with old_r, count[y] > 2 (so it stays >= 2 after loss).
        # If any such y has count[y] == 2, this change is bad.
        
        # So, for each prime p:
        # 1. new_r = x_target % p
        # 2. if new_r == residues[p_idx], skip (no change).
        # 3. Check if changing breaks prefix:
        #    Iterate through y in [1, x_target-1] where y % p == residues[p_idx].
        #    If any such y has count[y] == 2, this p is NOT a valid candidate.
        # 4. If valid, we can apply the change.
        
        # Which p to pick?
        # We want to maximize future coverage.
        # A good heuristic: Pick the smallest p that is valid.
        # Why? Small primes cover more numbers > x_target, potentially helping extend the prefix further.
        
        # Let's implement this.
        
        found_valid = False
        # Try primes in increasing order
        # But checking all 168 primes every iteration might be slow if we iterate through y for each.
        # x_target can be up to 10000.
        # For p=2, we check 5000 positions.
        # For p=997, we check 10 positions.
        # Total work per iteration ~ sum(x_target/p) ~ x_target * log(log(x_target)).
        # 10000 * ~3 = 30000 operations per iteration.
        # 50000 iterations * 30000 ops = 1.5 billion ops. Too slow in Python.
        
        # Optimization:
        # We don't need to check ALL y < x_target.
        # We only care about y where count[y] == 2.
        # Let criticals = [y for y in 1..x_target-1 if count[y] == 2]
        # For a candidate p, we just check if any y in criticals satisfies y % p == old_r.
        # If yes, invalid.
        
        # How to find criticals efficiently?
        # We can maintain a list or just find them on the fly?
        # Finding them is O(x_target).
        # Checking a candidate p against criticals is O(len(criticals) * 1) if we just iterate criticals.
        # But we need to check y % p == old_r.
        
        # Alternative:
        # For a candidate p, the positions covered are old_r, old_r+p, ...
        # We just need to check if ANY of these positions (that are < x_target) has count == 2.
        # Number of such positions is x_target // p.
        # So cost is O(x_target // p).
        # Sum over p of x_target // p is x_target * log(log x_target) ~ 30000.
        # This is acceptable in Python if optimized with NumPy?
        # No, looping over p in Python is slow.
        
        # Let's vectorize the "validity check" for ALL primes at once?
        # Hard to do because the condition depends on old_r and count.
        
        # Let's try a simpler heuristic:
        # Only try the first 20 primes (smallest). They have the highest impact.
        # If none of the first 20 primes can extend the prefix, maybe we're stuck or need a different strategy.
        # Actually, large primes are "cheap" to change (few positions affected), so they are safer.
        # But they don't help extend the prefix as much.
        # Let's try a mix: Try small primes first for impact, but if they fail, try larger ones for safety?
        # Actually, if a small prime fails, it's because it breaks a critical point.
        # A large prime might not cover any critical points, so it might be valid.
        
        # Let's try all primes, but break early if we find a valid one?
        # No, we want the BEST one.
        # Heuristic: Smallest p that is valid.
        
        # To speed up, we can pre-compute the "critical mask" for the current state?
        # critical_mask[y] = 1 if count[y] == 2 else 0.
        # For a prime p with old_r, the positions are old_r, old_r+p, ...
        # We need to check if np.any(critical_mask[old_r::p] * (np.arange(old_r, x_target, p) < x_target))
        
        # Let's implement a loop over primes, but optimize the inner check.
        
        # Actually, let's just try the first 50 primes. If we can't extend, we might be in a local minimum.
        # To escape, we could allow "bad" moves (simulated annealing style), but let's stick to greedy first.
        
        # Revised Plan:
        # 1. Identify x_target.
        # 2. For each prime p (in increasing order):
        #    a. new_r = x_target % p
        #    b. if new_r == residues[i], continue
        #    c. Check if change breaks prefix:
        #       - Get indices of positions covered by old_r in [1, x_target-1]
        #       - Check if any of these have count == 2
        #    d. If valid, apply change and break.
        # 3. If no valid prime found in first 50, try next 50? Or give up and random restart?
        
        # Let's try all 168 primes, but break on first valid one?
        # Or pick the one with smallest p?
        
        # Implementation detail:
        # To check if change breaks prefix:
        # positions = np.arange(old_r if old_r>0 else p, x_target, p)
        # if len(positions) > 0:
        #     if np.any(counts[positions] == 2):
        #         continue # Invalid
        
        # This is O(x_target/p) per prime.
        # Total O(x_target * sum(1/p)) ~ O(x_target * log log x_target).
        
        # Let's try this.
        
        applied = False
        # We'll iterate through primes. To make it deterministic but varied, we can shuffle?
        # No, greedy usually benefits from consistent ordering (small p first).
        
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if old_r == new_r:
                continue
                
            # Check validity
            # Positions covered by old_r in [1, x_target-1]
            start_pos = old_r if old_r > 0 else p
            if start_pos >= x_target:
                # No positions in [1, x_target-1] are covered by old_r
                # So change is safe for prefix
                valid = True
            else:
                positions = np.arange(start_pos, x_target, p)
                # Check if any of these have count == 2
                if np.any(counts[positions] == 2):
                    valid = False
                else:
                    valid = True
            
            if valid:
                # Apply change
                update_counts(i, old_r, new_r, -1) # Remove old
                residues[i] = new_r
                update_counts(i, new_r, new_r, 1)  # Add new (note: update_counts uses 'new_r' for +1, but we pass old_r as dummy? No, my update_counts signature is (idx, old_r, new_r, delta). For delta=1, it uses new_r. For delta=-1, it uses old_r. So this is correct.)
                
                applied = True
                break
        
        if not applied:
            # No valid move found to extend the prefix.
            # We are stuck.
            # Strategy: Try to improve coverage beyond x_target without breaking prefix?
            # Or, try a random move that doesn't break prefix, to change the landscape?
            # Or, accept that we can't extend further with this greedy strategy and stop?
            # If we stop, we might have a suboptimal solution.
            
            # Let's try a "repair" or "shuffle" strategy:
            # Pick a random prime, try a random residue, if it doesn't break prefix, keep it?
            # This might help escape local minima.
            
            # But if we can't even extend the prefix, random moves won't help extend it immediately.
            # They might allow a different path later.
            
            # For now, let's break if we can't extend.
            # Or, try larger primes? We already tried all.
            
            # Maybe we should try to maximize the count at x_target even if it breaks a non-critical point?
            # No, breaking a critical point reduces L.
            
            # Let's try a different heuristic:
            # What if we look at x_target + 1?
            # No, we must fix x_target first.
            
            # If we can't fix x_target, we are done.
            break
            
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
