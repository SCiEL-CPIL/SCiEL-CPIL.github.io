"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_primes(limit):
    """Return list of primes up to limit."""
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
    Returns the value L.
    """
    n = len(residues)
    if n == 0:
        return 0
    
    # We only need to check up to a reasonable bound to find the first failure.
    # The score is capped at 10^5, but we can stop early if we find a gap.
    # Let's estimate the max L we might reach. 
    # If we want L, we need 2 * L hits distributed among n primes.
    # Average hits per prime ~ 2L/n.
    # For n=168, if L=1000, avg hits ~ 12.
    # We can just compute the count array up to max_check or until we find the first failure.
    
    # To be efficient, we can compute the count for a window.
    # However, numpy boolean indexing is fast.
    # Let's compute counts for 1..max_check.
    # But 10^5 is small enough that we can just do it.
    
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for r, p in zip(residues, primes):
        # x = r mod p
        # The numbers are r, r+p, r+2p, ...
        # Note: r can be 0. If r=0, the numbers are p, 2p, ... (since we start counting from 1, 
        # 0 is not in 1..max_check, but p is).
        # Actually, if r=0, x % p == 0. The smallest positive x is p.
        # If r > 0, the smallest positive x is r.
        
        # Generate the array of indices
        # We need indices i such that 1 <= i <= max_check and i % p == r
        # This is equivalent to i = k*p + r
        # k*p + r >= 1 => k >= (1-r)/p
        # k*p + r <= max_check => k <= (max_check - r)/p
        
        # Start index:
        if r == 0:
            start = p
        else:
            start = r
            
        if start > max_check:
            continue
            
        # Create the sequence
        # np.arange(start, max_check + 1, p)
        # This might be slow if done in a loop for 168 primes, but 168 is small.
        # Let's try to vectorize or just use arange.
        idx = np.arange(start, max_check + 1, p)
        counts[idx] += 1
        
    # Find the first index i in 1..max_check where counts[i] < 2
    # We want the largest L such that counts[1..L] >= 2
    # So we look for the first i where counts[i] < 2.
    # If no such i exists in 1..max_check, return max_check (capped).
    
    # Vectorized check
    # We want the first index in 1..max_check where count < 2
    # Let's find all indices where count < 2
    failing_indices = np.where(counts[1:max_check+1] < 2)[0]
    
    if len(failing_indices) == 0:
        return max_check
    else:
        # The first failing index in the slice corresponds to 1 + failing_indices[0]
        # The score is the number of integers before this one.
        # If the first failure is at index i (1-based), then L = i - 1.
        # failing_indices[0] is 0-based in the slice, so the 1-based index is 1 + failing_indices[0].
        # L = (1 + failing_indices[0]) - 1 = failing_indices[0]
        return failing_indices[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Initial heuristic: 
    # We want to cover 1..L twice.
    # A good starting point is to assign residues such that they cluster around small numbers.
    # For each prime p, if we pick r_p = 1, it covers 1, 1+p, 1+2p...
    # If we pick r_p = 2, it covers 2, 2+p...
    # To maximize coverage of small numbers, we should pick residues that hit small numbers.
    
    # Strategy: 
    # 1. Initialize residues with a simple pattern.
    # 2. Compute current score L.
    # 3. Identify the "bottleneck" integer x = L+1 which has count < 2.
    # 4. Try to change residues of primes to increase the count at x and nearby integers,
    #    without decreasing the count at integers <= L below 2.
    
    # Initial assignment: 
    # Let's try to make the first 2*168 = 336 numbers covered.
    # A simple greedy: for each prime, pick the residue that hits the most currently under-covered small numbers.
    
    # Let's start with a "random" or "sequential" assignment and then optimize.
    # Better: Start with r_p = 1 for all p. This covers 1 heavily, but 2,3.. less.
    # Even better: Distribute residues to cover 1..K roughly evenly.
    
    # Let's use a greedy construction first.
    # We want to build the array of residues.
    # For each prime p (in increasing order), choose r_p to maximize coverage of the prefix.
    
    # Actually, a simpler and effective heuristic for this specific "doubly covered" problem:
    # The seed used (i+2)//2 % p.
    # Let's try a different initialization: 
    # Assign residues to cover the range [1, 2*n_primes] as evenly as possible.
    
    # Let's try a local search from a few different starting points.
    
    best_score = 0
    best_residues = None
    
    # Starting Point 1: The seed
    res1 = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    s1 = _compute_score(res1, primes)
    if s1 > best_score:
        best_score = s1
        best_residues = res1[:]
        
    # Starting Point 2: All 1s
    res2 = [1 % p for p in primes]
    s2 = _compute_score(res2, primes)
    if s2 > best_score:
        best_score = s2
        best_residues = res2[:]
        
    # Starting Point 3: Random
    np.random.seed(42)
    res3 = [np.random.randint(0, p) for p in primes]
    s3 = _compute_score(res3, primes)
    if s3 > best_score:
        best_score = s3
        best_residues = res3[:]
        
    # Starting Point 4: Greedy Fill
    # Try to assign residues to cover 1..N where N is around 200.
    # For each prime, pick the residue that hits the most numbers in 1..N that are currently covered 0 or 1 times.
    
    # Let's run a local search on the best candidate found so far.
    # We will use the best_residues as the starting point for hill climbing.
    
    current_residues = best_residues[:]
    current_score = best_score
    
    # Precompute the count array for the current state to speed up evaluation?
    # Re-evaluating the full score is O(N * P) which is 168 * 100000 = 16.8M ops.
    # In Python, this might take ~1-2 seconds.
    # If we do 50 iterations, that's 50-100 seconds. Too slow.
    # We need a faster way to estimate the impact of a change.
    
    # Alternative: Only check a window around the current score.
    # If current score is L, we only care about integers near L.
    # Changing a residue r_p to r'_p affects the counts at x where x % p == r_p (decrease) and x % p == r'_p (increase).
    # We can maintain the count array incrementally.
    
    # Let's implement an incremental scorer.
    # State: counts array (size 100001)
    # When changing residue for prime p from old_r to new_r:
    # 1. Subtract 1 from counts at indices i where i % p == old_r
    # 2. Add 1 to counts at indices i where i % p == new_r
    # 3. Check if the new minimum count in the prefix is still >= 2.
    
    # This is O(L/p) per update. Sum over p is L * sum(1/p) ~ L * log(log(N)).
    # For L=1000, this is very fast.
    
    # Let's re-initialize counts for best_residues
    max_check = 100000
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(current_residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            idx = np.arange(start, max_check + 1, p)
            counts[idx] += 1
            
    # Find current L
    # We know current_score, but let's verify.
    # failing_indices = np.where(counts[1:max_check+1] < 2)[0]
    # current_score = failing_indices[0] if len(failing_indices)>0 else max_check
    
    # Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find the current bottleneck
        # We need the first index i where counts[i] < 2
        # Since we only changed a few things, the bottleneck might have moved slightly.
        # But to be safe, let's find it.
        # Optimization: Keep track of the current L.
        # If we only improved, L stays same or increases.
        # If we made a mistake, we revert.
        
        # Let's try to improve the score by targeting the first failing integer.
        # Find first failing index in 1..max_check
        # This is O(max_check) per iteration. 100k is fine.
        failing_indices = np.where(counts[1:max_check+1] < 2)[0]
        if len(failing_indices) == 0:
            L = max_check
        else:
            L = failing_indices[0]
            
        if L == max_check:
            break
            
        target_x = L + 1
        
        # Try changing each prime's residue to see if we can increase the count at target_x
        # or more generally, extend the prefix.
        # A better move: Change a residue to cover target_x AND another nearby uncovered/under-covered number.
        
        # For each prime i:
        #   Current residue r_i.
        #   Target residue r_new such that target_x % p_i == r_new.
        #   This will increase count at target_x by 1.
        #   It will decrease count at old positions (where x % p_i == r_i) by 1.
        #   We must ensure that no position y <= L has its count drop below 2.
        
        best_gain = 0
        best_move = None # (prime_idx, new_r)
        
        for i, p in enumerate(primes):
            r_old = current_residues[i]
            r_new = target_x % p
            
            if r_new == r_old:
                continue
                
            # Check impact
            # 1. Identify positions that will lose coverage: x where x % p == r_old
            # 2. Identify positions that will gain coverage: x where x % p == r_new
            
            # We only care about positions 1..L (must stay >= 2) and positions > L (want to increase)
            
            # Positions losing coverage:
            # If r_old == 0, start = p. Else start = r_old.
            # We need to check if any of these positions y in 1..L has counts[y] == 2.
            # If so, making this change would drop it to 1, which is bad (unless we also add coverage elsewhere? No, this is a single change).
            # So, if ANY y in 1..L with counts[y] == 2 is in the "losing" set, this move is INVALID (unless it also covers that y? No, r_new != r_old).
            
            # Let's check the "losing" set in 1..L.
            if r_old == 0:
                start_old = p
            else:
                start_old = r_old
                
            # Iterate through losing positions in 1..L
            # We can do this by stepping through p.
            # But we need to check counts.
            
            # Optimization: 
            # Precompute for each prime p, the list of positions in 1..L where counts[pos] == 2.
            # If any of these positions is congruent to r_old mod p, the move is invalid.
            
            # This is getting complex. Let's just simulate the change on a small window or use numpy.
            
            # Create a temporary copy of counts? No, too slow.
            # Let's check validity:
            # Valid if: for all y in 1..L, if y % p == r_old, then counts[y] > 2.
            
            # Let's generate the losing indices in 1..L
            if start_old <= L:
                losing_idx = np.arange(start_old, L + 1, p)
                # Check if any of these have count == 2
                if np.any(counts[losing_idx] == 2):
                    continue # Invalid move
            
            # If valid, calculate the gain.
            # The gain is how many new integers > L become covered (count >= 2) or how far the prefix extends.
            # Actually, we just want to maximize the new L.
            # If we fix target_x, the new L is at least target_x.
            # It might extend further if target_x+1, target_x+2... also get fixed.
            
            # Let's estimate the new L by checking the counts after the hypothetical change.
            # We can't easily update the global counts array without modifying it.
            # Let's just pick the move that fixes the most "critical" spots.
            
            # Heuristic: Prefer moves that also cover target_x + 1, target_x + 2, etc.
            # Or simply: If the move is valid, it's a candidate.
            # We can evaluate the "quality" by seeing how many of the next K integers are covered by the new residue.
            
            # Let's just take the first valid move? Or the one that covers the most additional points in 1..L+10?
            
            # Let's check if the new residue covers target_x + 1, target_x + 2...
            # This is getting complicated. Let's just try the move if it's valid.
            # To choose between multiple valid moves, pick the one that maximizes the count at target_x (which is always +1) 
            # and minimizes the "damage" (though we already checked damage is 0).
            
            # Actually, different primes have different periods.
            # A move on a small prime affects many points. A move on a large prime affects few.
            # We should prefer moves on larger primes? No, larger primes are sparser.
            # Moving a small prime residue affects the counts of many numbers.
            # But we already checked that it doesn't drop any count to 1.
            
            # Let's just pick the first valid move we find.
            best_move = (i, r_new)
            break # Take the first valid move to save time? 
            # Or we can collect a few and pick the best.
            # Given time constraints, taking the first valid move is a reasonable heuristic for hill climbing.
            
        if best_move is not None:
            i, r_new = best_move
            p = primes[i]
            r_old = current_residues[i]
            
            # Apply the change
            # Subtract from old
            if r_old == 0:
                start_old = p
            else:
                start_old = r_old
            if start_old <= max_check:
                idx_old = np.arange(start_old, max_check + 1, p)
                counts[idx_old] -= 1
                
            # Add to new
            if r_new == 0:
                start_new = p
            else:
                start_new = r_new
            if start_new <= max_check:
                idx_new = np.arange(start_new, max_check + 1, p)
                counts[idx_new] += 1
                
            current_residues[i] = r_new
            
            # Check if score improved
            # Find new L
            failing_indices_new = np.where(counts[1:max_check+1] < 2)[0]
            new_L = failing_indices_new[0] if len(failing_indices_new) > 0 else max_check
            
            if new_L > current_score:
                current_score = new_L
                best_score = current_score
                best_residues = current_residues[:]
                improved = True
            else:
                # Revert
                # Add back to old
                if r_old == 0:
                    start_old = p
                else:
                    start_old = r_old
                if start_old <= max_check:
                    idx_old = np.arange(start_old, max_check + 1, p)
                    counts[idx_old] += 1
                    
                # Subtract from new
                if r_new == 0:
                    start_new = p
                else:
                    start_new = r_new
                if start_new <= max_check:
                    idx_new = np.arange(start_new, max_check + 1, p)
                    counts[idx_new] -= 1
                    
                current_residues[i] = r_old
                
        # If no valid move found, we are stuck.
        # We could try a random perturbation (simulated annealing style) or break.
        # For now, if no move improves, we break to avoid infinite loops.
        # But maybe a non-improving move leads to a better one later?
        # Given the time limit, let's try a few random restarts if stuck.
        
        # Let's add a counter for stuck iterations.
        # If stuck, pick a random prime and random residue, apply, and see if it helps later.
        # This is risky. Let's just break for now to be safe.
        
        # Actually, if no move improves the score, hill climbing has converged to a local optimum.
        # We can try a random restart from a different initial state.
        # But we have limited time.
        
        # Let's try one random perturbation if stuck.
        # Pick a random prime, change residue to a random value.
        # If it improves, great. If not, revert.
        # Repeat a few times.
        
        # To keep it simple and fast, let's just break if no improving move is found.
        break

    return best_residues
# EVOLVE-BLOCK-END
