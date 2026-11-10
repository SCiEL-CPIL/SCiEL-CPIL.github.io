import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit (using 110s to be safe)
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = False
        return np.array([i for i, is_prime in enumerate(sieve) if is_prime], dtype=np.int64)
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # We will work with a window of size 2000 initially, expanding if needed.
    # The score is the largest L such that every x in 1..L has count >= 2.
    # Let's use a window of 2000 for the main search logic to keep memory/time manageable.
    # If we reach 2000, we can expand, but 1000 is the target.
    WINDOW = 2000
    x_vals = np.arange(1, WINDOW + 1, dtype=np.int64)
    
    # Initialize residues to 0
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Precompute masks for each prime? 
    # masks[i] is a boolean array of length WINDOW where masks[i][j] is True if x_vals[j] % primes[i] == residues[i]
    # Actually, residues change. 
    # Let's precompute the "base" modulo structure.
    # For prime p, the set of x in 1..WINDOW such that x % p == r is periodic.
    
    # Efficient representation:
    # counts[x] = sum over i of (1 if x % p_i == r_i else 0)
    
    # We maintain `counts` as a numpy array of length WINDOW.
    counts = np.zeros(WINDOW, dtype=np.int32)
    
    # Helper to update counts when residue of prime i changes from r_old to r_new
    def update_counts(i, r_old, r_new):
        p = primes[i]
        if r_old == r_new:
            return
        # Remove old
        # x % p == r_old  =>  x in {r_old, r_old+p, ...} within 1..WINDOW
        # Note: x starts from 1.
        # If r_old == 0, x is multiple of p.
        # Indices in x_vals (1-indexed values)
        # x_vals is [1, 2, ..., WINDOW]
        # We need to find indices j such that x_vals[j] % p == r_old
        
        # Vectorized approach for update:
        # Create a boolean mask for the specific residue
        # mask_old = (x_vals % p == r_old)
        # mask_new = (x_vals % p == r_new)
        # counts -= mask_old.astype(np.int32)
        # counts += mask_new.astype(np.int32)
        
        # This is O(WINDOW) per update.
        # 2000 ops per update. Very fast.
        
        mask_old = (x_vals % p == r_old).astype(np.int32)
        mask_new = (x_vals % p == r_new).astype(np.int32)
        counts -= mask_old
        counts += mask_new

    # Helper to compute the score (first failure index)
    def get_score():
        # Find first index where counts < 2
        # If all >= 2, return WINDOW
        fail = counts < 2
        if not np.any(fail):
            return WINDOW
        first_fail_idx = np.argmax(fail)
        # x_vals is 1-indexed. Index 0 corresponds to x=1.
        # If fail[0] is True, x=1 fails, so L=0.
        # If fail[k] is True, x=k+1 fails, so L=k.
        return first_fail_idx

    current_score = get_score()
    
    # Strategy:
    # 1. Greedy Phase: 
    #    While there is a deficit (score < WINDOW):
    #    a. Find the first x in 1..score+1 that has count < 2. (Actually score+1 is the first failing x)
    #    b. We want to increase the count of this x.
    #    c. To do so, we must pick a prime p_i and set r_i = x % p_i.
    #    d. Which prime to pick? 
    #       - It must be a prime that currently does NOT cover x (i.e., x % p_i != r_i). 
    #         If it already covers x, changing its residue would remove coverage, which is bad unless we add 2 elsewhere? 
    #         No, we need count >= 2. If count is 1, we need +1. 
    #         If count is 0, we need +2.
    #       - Choosing a large prime is generally "safer" because it covers fewer other numbers in the window, 
    #         reducing the risk of breaking other numbers' coverage (specifically, it's less likely to collide with 
    #         requirements of other numbers, or rather, its "footprint" is sparse).
    #       - However, we must ensure that setting r_i = x % p_i doesn't break the coverage of other numbers 
    #         that are currently at exactly 2 and rely on this prime.
    #       
    #    Let's refine the greedy step:
    #    1. Identify the "bottleneck" integer x_b = current_score + 1.
    #    2. Get its current count c_b.
    #    3. We need to add (2 - c_b) coverage to x_b.
    #    4. Candidate primes: Those p_i such that x_b % p_i != r_i (currently not covering x_b).
    #       If we set r_i = x_b % p_i, we add 1 to x_b's count.
    #       We also affect other numbers. Specifically, for any y in 1..WINDOW, if y % p_i == r_i_new, 
    #       its count increases by 1. If y % p_i == r_i_old, its count decreases by 1.
    #    5. To minimize damage, we should pick a prime p_i that:
    #       a) Is large (sparse coverage).
    #       b) Currently covers as few "critical" numbers (those with count == 2) as possible.
    #          Actually, if we change r_i, we remove coverage from all y where y % p_i == r_i_old.
    #          If any of those y have count == 2, they will drop to 1 and become new bottlenecks (or worse).
    #          So, a good heuristic is to pick a prime p_i that currently covers NO numbers with count == 2, 
    #          or minimally covers them.
    #       c) Ideally, we pick the largest such prime.
    
    # Let's implement a more robust local search hybrid.
    # The greedy approach might get stuck if no "safe" prime exists.
    
    # Phase 1: Greedy repair
    # We will iterate.
    
    # To speed up "which primes cover critical numbers", we can just evaluate the change.
    # Since WINDOW is 2000, checking the impact of a move is O(WINDOW).
    # Number of primes is 168.
    # Checking all 168 primes for the bottleneck is 168 * 2000 = 336,000 ops.
    # This is very fast in NumPy if vectorized, or even in Python loop with precomputed masks?
    # Precomputing masks for all primes and all residues is 168 * 1000 * 2000 booleans = 336 MB. Too much.
    # But we can compute mask on the fly: (x_vals % p == r).
    
    improved = True
    max_iterations = 5000 # Safety
    
    while improved and time.time() < deadline and current_score < WINDOW:
        improved = False
        iterations_this_phase = 0
        
        while improved and time.time() < deadline and current_score < WINDOW and iterations_this_phase < 100:
            iterations_this_phase += 1
            x_b = current_score + 1
            c_b = counts[x_b - 1] # 0-indexed
            
            if c_b >= 2:
                # This shouldn't happen if get_score is correct, but just in case
                current_score = get_score()
                continue
                
            # We need to add (2 - c_b) coverage.
            # Let's try to find a prime to switch.
            # Candidates: all primes i.
            # For each prime i:
            #   r_new = x_b % primes[i]
            #   r_old = residues[i]
            #   If r_new == r_old, it already covers x_b. 
            #      If c_b < 2, this is a problem? No, if it covers it, it's already counted.
            #      So if r_new == r_old, changing it to something else would REMOVE coverage. Bad.
            #      So we only consider primes where r_new != r_old.
            
            best_move = None
            best_score_after = current_score
            best_i = -1
            best_r_new = -1
            
            # Heuristic: Sort primes by size descending? 
            # Larger primes are "sparser".
            # Let's iterate all primes, but prioritize larger ones in tie-breaking.
            
            for i in range(n_primes):
                p = primes[i]
                r_new = x_b % p
                r_old = residues[i]
                
                if r_new == r_old:
                    continue
                
                # Simulate the change
                # We need to check if this change keeps all x in 1..current_score covered >= 2.
                # And ideally extends the score.
                
                # Impact:
                # 1. x_b gains 1 coverage.
                # 2. All y such that y % p == r_old LOSE 1 coverage.
                # 3. All y such that y % p == r_new GAIN 1 coverage.
                
                # Critical check: 
                # Does any y in 1..current_score have count == 2 and y % p == r_old?
                # If yes, it drops to 1, breaking the prefix.
                
                # Let's find the set of y covered by r_old for prime p.
                # y_old_mask = (x_vals % p == r_old)
                # We care about y in 1..current_score.
                
                # Efficient check:
                # Create a mask for the range 1..current_score
                # Actually, we can just compute the new counts for the relevant range?
                # Or simpler: 
                # 1. Identify indices in 1..current_score where count == 2 and x % p == r_old.
                #    If any exist, this move is "unsafe" for the prefix (it will break it).
                #    However, if x_b is the *first* failure, breaking a later part of the prefix 
                #    (index > current_score) is not a problem for the *current* score, 
                #    but it might prevent future growth. 
                #    Breaking a part of the prefix (index <= current_score) is fatal for the score.
                
                # So, condition for "Safe" move:
                # No y in 1..current_score has (counts[y-1] == 2 AND x_vals[y-1] % p == r_old).
                
                # Let's check this.
                # y_range = x_vals[:current_score]
                # counts_prefix = counts[:current_score]
                # critical_mask = (counts_prefix == 2)
                # old_cover_mask = (y_range % p == r_old)
                # if np.any(critical_mask & old_cover_mask):
                #     continue # Unsafe
                
                # This check is O(current_score) per prime.
                # 168 primes * 2000 = 336k ops.
                # This is fine.
                
                y_range = x_vals[:current_score]
                counts_prefix = counts[:current_score]
                
                # Optimization: Precompute counts_prefix == 2 once outside the loop?
                # It changes if we make a move. But we are just simulating.
                
                critical_mask = (counts_prefix == 2)
                if not np.any(critical_mask):
                    # No critical numbers in prefix. Any move is safe for the prefix.
                    # Pick the largest prime?
                    # Let's just pick the first valid one, but prefer large p.
                    # We'll collect candidates and pick best.
                    pass
                else:
                    old_cover_mask = (y_range % p == r_old)
                    if np.any(critical_mask & old_cover_mask):
                        continue # This move breaks the prefix
                
                # If we reach here, the move is "safe" for the current prefix.
                # Now, does it improve the score?
                # It definitely increases count of x_b by 1.
                # If c_b was 1, it becomes 2. The prefix extends past x_b.
                # If c_b was 0, it becomes 1. The prefix does NOT extend past x_b yet.
                
                # We want to maximize the new score.
                # If c_b == 1, the new score will be at least current_score + 1.
                # It might extend further if x_b+1 also had count 1 and gained coverage?
                # Unlikely to gain coverage from this specific move unless x_b+1 % p == r_new.
                
                # Let's estimate the new score.
                # We can simulate the counts change for the whole window?
                # That's O(WINDOW). 2000 ops.
                # 168 * 2000 = 336k ops per iteration.
                # If we do 100 iterations, 33M ops. Fast.
                
                # Let's do a full simulation to get the exact new score.
                # This is more robust.
                
                # Simulate:
                # new_counts = counts.copy()
                # new_counts -= (x_vals % p == r_old).astype(np.int32)
                # new_counts += (x_vals % p == r_new).astype(np.int32)
                # new_score = get_score_from_array(new_counts)
                
                # To avoid copying 2000 ints 168 times (336k ints copy), 
                # we can just compute the delta and apply to a temp view? 
                # Or just copy. 2000 * 4 bytes = 8KB. 168 copies = 1.3MB. Trivial.
                
                mask_old = (x_vals % p == r_old).astype(np.int32)
                mask_new = (x_vals % p == r_new).astype(np.int32)
                
                temp_counts = counts - mask_old + mask_new
                
                # Check if prefix is still valid (should be, by our safety check, but verify)
                # Actually, the safety check only looked at count==2. 
                # What if count was 1 and dropped to 0? 
                # If count was 1 in the prefix, it's not a "failure" for the current score L, 
                # because L is the *largest* such that *all* are >= 2. 
                # Wait. If x in 1..L has count 1, then L is not valid. 
                # So by definition, all x in 1..current_score have count >= 2.
                # So if we drop a count from 2 to 1, it becomes a failure.
                # If we drop a count from 3 to 2, it's still valid.
                # So checking count==2 is sufficient for "breaking" the prefix.
                
                # However, we also need to make sure we didn't break something else?
                # No, the definition of score is strictly the first failure.
                
                new_score = get_score_from_counts(temp_counts)
                
                if new_score > best_score_after:
                    best_score_after = new_score
                    best_i = i
                    best_r_new = r_new
                    # We can break early if we find a very good move?
                    # Or just find the best.
                    # Let's find the best to be safe.
                    
            if best_i != -1:
                # Apply the best move
                p = primes[best_i]
                r_old = residues[best_i]
                residues[best_i] = best_r_new
                update_counts(best_i, r_old, best_r_new)
                current_score = best_score_after
                improved = True
            else:
                # No safe move found in greedy phase.
                # Break out to local search or stop.
                break

        # Phase 2: Local Search / Hill Climbing
        # If greedy stuck, try random or systematic local search.
        # We will try to change residues of primes that cover the current bottleneck x_b.
        # Or just do a general hill climbing pass on the current best.
        
        # Let's do a simple hill climbing pass over all primes, trying all residues.
        # But only if we have time.
        
        if time.time() < deadline - 10: # Leave 10s for finishing
             # Hill Climbing
             hc_improved = True
             while hc_improved and time.time() < deadline - 5:
                 hc_improved = False
                 for i in range(n_primes):
                     p = primes[i]
                     r_old = residues[i]
                     
                     # Try all residues
                     best_r = r_old
                     best_s = current_score
                     
                     # Optimization: Only try residues that cover x_b? 
                     # No, other moves might help by freeing up constraints?
                     # Let's try all, but vectorize the score calculation?
                     # We can't easily vectorize get_score.
                     
                     # But we can speed up the check.
                     # For each r in 0..p-1:
                     #   Simulate.
                     
                     # This is O(p * WINDOW) per prime.
                     # Sum p * WINDOW = 76127 * 2000 = 152M ops.
                     # In Python loop, this is too slow.
                     # In NumPy, we can batch?
                     
                     # Let's just do a random subset of residues if p is large?
                     # Or just skip hill climbing if too slow?
                     
                     # Given the greedy phase is strong, maybe we don't need heavy HC.
                     # Let's just try a few random moves or the "best" move from a quick scan.
                     
                     # Quick scan: Try residues that are "popular" or random.
                     # Let's just try 10 random residues per prime.
                     if p > 100:
                         candidates = np.random.choice(p, 10, replace=False)
                     else:
                         candidates = np.arange(p)
                         
                     for r_new in candidates:
                         if r_new == r_old:
                             continue
                         mask_old = (x_vals % p == r_old).astype(np.int32)
                         mask_new = (x_vals % p == r_new).astype(np.int32)
                         temp_counts = counts - mask_old + mask_new
                         s = get_score_from_counts(temp_counts)
                         if s > best_s:
                             best_s = s
                             best_r = r_new
                             
                     if best_r != r_old:
                         update_counts(i, r_old, best_r)
                         residues[i] = best_r
                         current_score = best_s
                         hc_improved = True
                         improved = True # Signal outer loop to restart greedy if needed? 
                                         # Actually, if HC improves, greedy might find more.
                                         # But we are in a while loop. 
                                         # Let's just let the outer loop handle it.
                                         # Wait, the outer loop condition is `improved`.
                                         # If HC improves, `improved` is True, so outer loop continues.
                                         # Then greedy runs again. Good.

    # Final check: Expand window if we hit the limit?
    # If current_score == WINDOW, we might be able to go higher.
    # But the task asks for max L.
    # If we reached 2000, we should check further.
    # However, the computation cost doubles.
    # Given the 120s limit, and we've used some time, let's just return the current best.
    # The score is capped at 10^5 in the problem description, but practically 
    # finding > 2000 is hard. 
    # If we are at 2000, we can try to extend to 3000 quickly if time permits.
    
    if current_score >= WINDOW and time.time() < deadline - 20:
        # Extend window to 3000
        WINDOW = 3000
        x_vals = np.arange(1, WINDOW + 1, dtype=np.int64)
        
        # Rebuild counts
        counts = np.zeros(WINDOW, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            counts += (x_vals % p == r).astype(np.int32)
            
        current_score = get_score_from_counts(counts)
        
        # Run greedy again on the new window?
        # Just a few iterations.
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            if current_score >= WINDOW:
                break
            x_b = current_score + 1
            # ... same greedy logic ...
            # For brevity and safety, we'll just do a quick hill climb on the new window
            # or just return.
            # Let's just do a quick pass.
            
            # Actually, let's just re-run the main greedy loop logic briefly.
            # To avoid code duplication, we'd need to refactor.
            # Given time constraints, let's just return the best found.
            # The probability of jumping from 2000 to 3000 without re-optimizing is low.
            # But if we are at 2000, we are already very good.
            
            # Let's do a simple local repair for the new bottleneck.
            x_b = current_score + 1
            best_move = None
            best_s = current_score
            for i in range(n_primes):
                p = primes[i]
                r_new = x_b % p
                r_old = residues[i]
                if r_new == r_old:
                    continue
                mask_old = (x_vals % p == r_old).astype(np.int32)
                mask_new = (x_vals % p == r_new).astype(np.int32)
                temp_counts = counts - mask_old + mask_new
                s = get_score_from_counts(temp_counts)
                if s > best_s:
                    best_s = s
                    best_move = (i, r_new)
            if best_move:
                i, r_new = best_move
                r_old = residues[i]
                update_counts(i, r_old, r_new)
                residues[i] = r_new
                current_score = best_s
            else:
                break

    return [int(r) for r in residues]

def get_score_from_counts(c):
    # c is a numpy array of counts for x=1..len(c)
    fail = c < 2
    if not np.any(fail):
        return len(c)
    first_fail_idx = np.argmax(fail)
    return first_fail_idx
