import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
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
    
    # Strategy:
    # 1. Start with a random configuration.
    # 2. Use a "Gap Repair" heuristic:
    #    - Find the first integer x in [1, Limit] that is not covered by at least 2 primes.
    #    - Identify the set of primes that *could* cover x (i.e., all of them, since we can change any residue).
    #    - Actually, more specifically:
    #      Let S_x be the set of primes p such that changing r_p to x % p would increase the coverage count of x by 1 (i.e., currently x != r_p mod p).
    #      Let T_x be the set of primes p such that currently x == r_p mod p.
    #      We need count(x) >= 2.
    #      If count(x) == 0, we need to add 2 covers.
    #      If count(x) == 1, we need to add 1 cover.
    #      
    #      A simpler approach for the "first failure" x_fail:
    #      We want to increase count(x_fail).
    #      We can pick a prime p not currently covering x_fail and set r_p = x_fail % p.
    #      This increases count(x_fail) by 1.
    #      However, this might break coverage for other numbers.
    #      Specifically, it removes coverage for the number x_old = r_p + k*p (the previous residue class).
    #      
    #      Better Strategy: "Two-Point" or "Swap" based on the first failure.
    #      Let x_fail be the first number with count < 2.
    #      We need to boost its count.
    #      Let's identify primes p that are "cheap" to switch to cover x_fail.
    #      A prime p is "cheap" if its current residue r_p covers a number y > x_fail (so we don't break the prefix) OR if y is already covered by >= 2 other primes (so we don't break the prefix).
    #      
    #      Let's implement a repair loop:
    #      1. Compute counts for x in 1..Limit.
    #      2. Find first x_fail with count < 2.
    #      3. If none, return success (or extend limit).
    #      4. We need to add coverage to x_fail.
    #         Candidate primes: All p.
    #         For each p, let r_new = x_fail % p.
    #         Let r_old = residues[p].
    #         If r_old == r_new, skip.
    #         The change affects:
    #            - x_fail: count increases by 1.
    #            - x_old = r_old + k*p: count decreases by 1.
    #         We want to choose p such that:
    #            a) x_fail count becomes >= 2 (needs 1 or 2 boosts depending on current count).
    #            b) x_old does NOT become < 2.
    #               i.e., count(x_old) - 1 >= 2  => count(x_old) >= 3.
    #               OR x_old > x_fail (so it's outside the current prefix, doesn't matter for L).
    #      
    #      If we find such a p, we make the move.
    #      If count(x_fail) was 0, we need two such moves (or one move that adds 2? No, each prime adds 1).
    #      So if count(x_fail) == 0, we need to find 2 primes p1, p2 such that moving them to cover x_fail is "safe".
    #      
    #      This is a structural change from hill climbing: it targets specific failures directly.

    Limit = 2000  # We check up to this limit for the score
    x_vals = np.arange(1, Limit + 1, dtype=np.int64)
    
    # Initialize residues randomly
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    def compute_counts(res_vec):
        counts = np.zeros(Limit, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = res_vec[i]
            counts += ((x_vals - r) % p == 0).astype(np.int32)
        return counts

    def get_score(counts):
        fail = counts < 2
        if not np.any(fail):
            return Limit
        first_fail_idx = np.argmax(fail)
        return x_vals[first_fail_idx] - 1

    counts = compute_counts(residues)
    current_score = get_score(counts)
    best_residues = residues.copy()
    best_score = current_score
    
    # Precompute for each prime p, the index in x_vals where it hits for a given residue?
    # No, that's hard.
    
    # Helper to check if a move is "safe"
    # Move: change residue of prime i from r_old to r_new.
    # This adds coverage to x_new (which is r_new mod p, but we care about specific x in 1..Limit)
    # Specifically, for the target x_fail, we set r_new = x_fail % p.
    # The "old" covered x in 1..Limit was x_old = r_old + k*p.
    # There is exactly one such x_old in 1..Limit if r_old < p and Limit >= p?
    # Not necessarily one. If Limit is large, there are multiple.
    # Wait, x % p == r is an arithmetic progression.
    # So changing r_old to r_new affects ALL x in 1..Limit such that x % p == r_old (decreases count)
    # and ALL x in 1..Limit such that x % p == r_new (increases count).
    
    # This makes the "safety" check more complex.
    # If we change r_p, we subtract the mask for r_old and add the mask for r_new.
    # We need to ensure that for all x in 1..current_score, the new count is >= 2.
    # And specifically, we want to fix x_fail.
    
    # Let's refine the repair strategy:
    # 1. Find x_fail.
    # 2. We need to increase count(x_fail).
    # 3. Try candidates for p.
    # 4. For a candidate p, calculate the delta counts:
    #    delta = mask(r_new) - mask(r_old)
    #    new_counts = counts + delta
    # 5. Check if new_counts[0:x_fail+1] are all >= 2. (Note: x_fail is 0-indexed in array)
    #    Actually, we only care that the prefix up to x_fail is valid.
    #    If valid, and count(x_fail) increased, we accept.
    
    # Optimization:
    # Instead of trying all p, we can prioritize p where r_old is "far" from the prefix?
    # Or just try all p, but limit the number of attempts per x_fail to avoid O(N^2) if stuck.
    
    # Let's use a "Greedy Repair" loop:
    # While time remains:
    #   1. Compute counts.
    #   2. Find x_fail.
    #   3. If none, break (or increase Limit? No, Limit is fixed for score).
    #   4. We need to fix x_fail.
    #   5. Generate a list of candidate primes.
    #      - Randomly shuffle primes.
    #   6. For each p in shuffled primes:
    #      - r_new = x_fail % p
    #      - r_old = residues[p]
    #      - If r_old == r_new, continue.
    #      - Calculate new_counts = counts - mask(r_old) + mask(r_new)
    #      - Check if min(new_counts[0:x_fail_idx+1]) >= 2.
    #      - If yes, accept move, update counts/residues, break inner loop.
    #   7. If no move accepted, try a "random kick":
    #      - Pick a random prime, change its residue randomly.
    #      - Or, accept a move that doesn't fully fix x_fail but improves some other metric?
    #      - Better: If stuck, do a random restart or a few random moves to escape local minima.
    
    # To speed up the check "min(new_counts[0:x_fail_idx+1]) >= 2":
    # We can precompute the prefix minimum of counts?
    # No, counts change.
    # But we can just check the slice. x_fail_idx is small (hopefully).
    # If x_fail is large, the slice is large.
    # However, if x_fail is large, we are close to the limit.
    
    # Let's implement this.
    
    improved = True
    while time.time() < deadline:
        counts = compute_counts(residues)
        current_score = get_score(counts)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
        
        if current_score >= Limit:
            break
            
        # Find first failure
        fail = counts < 2
        if not np.any(fail):
            break
        x_fail_idx = np.argmax(fail)
        x_fail = x_vals[x_fail_idx]
        
        # We need to increase count(x_fail).
        # Current count is c_fail = counts[x_fail_idx].
        # We need c_fail + 1 >= 2 => c_fail >= 1.
        # If c_fail == 0, we need 2 moves.
        # If c_fail == 1, we need 1 move.
        
        moves_needed = 2 - c_fail
        
        if moves_needed > 2:
            # This shouldn't happen if we start from valid or repair incrementally,
            # but if random init is bad, counts can be 0.
            # We'll just try to add 1 cover per iteration.
            moves_needed = 1
            
        fixed = False
        # Shuffle primes for randomization
        perm = np.random.permutation(n_primes)
        
        for idx in perm:
            if fixed:
                break
            i = idx
            p = primes[i]
            r_old = residues[i]
            r_new = x_fail % p
            
            if r_old == r_new:
                continue
                
            # Calculate delta
            # mask_old = (x_vals - r_old) % p == 0
            # mask_new = (x_vals - r_new) % p == 0
            # new_counts = counts - mask_old + mask_new
            
            # Optimization: Only check the prefix up to x_fail_idx
            # We need new_counts[0:x_fail_idx+1] >= 2
            
            # Let's compute the masks for the prefix only?
            # x_prefix = x_vals[0:x_fail_idx+1]
            # mask_old_prefix = (x_prefix - r_old) % p == 0
            # mask_new_prefix = (x_prefix - r_new) % p == 0
            # new_counts_prefix = counts[0:x_fail_idx+1] - mask_old_prefix + mask_new_prefix
            # if np.min(new_counts_prefix) >= 2:
            #    accept
            
            # But we also need to update the full counts if accepted.
            # So we might as well compute full masks?
            # Or compute prefix masks for check, and if accepted, update full counts.
            
            # Let's try prefix check first.
            x_prefix = x_vals[0:x_fail_idx+1]
            
            # Vectorized check
            # Note: x_prefix is int64. r_old, r_new are int.
            # (x_prefix - r_old) % p
            # This is fast for small x_fail_idx.
            
            # If x_fail_idx is large (e.g., 1000), this is 1000 ops. Very fast.
            
            mask_old_prefix = ((x_prefix - r_old) % p == 0).astype(np.int32)
            mask_new_prefix = ((x_prefix - r_new) % p == 0).astype(np.int32)
            
            new_counts_prefix = counts[0:x_fail_idx+1] - mask_old_prefix + mask_new_prefix
            
            if np.min(new_counts_prefix) >= 2:
                # Check if this move actually increases count at x_fail
                # It should, by construction (r_new = x_fail % p, so x_fail is in mask_new_prefix)
                # And x_fail was not in mask_old_prefix (since r_old != r_new)
                # So count increases by 1.
                
                # Accept move
                # Update full counts
                mask_old_full = ((x_vals - r_old) % p == 0).astype(np.int32)
                mask_new_full = ((x_vals - r_new) % p == 0).astype(np.int32)
                
                counts -= mask_old_full
                counts += mask_new_full
                residues[i] = r_new
                
                # If we still need more moves (e.g., c_fail was 0), we loop again?
                # The outer while loop will recompute and find the next failure.
                # But if x_fail still has count < 2, it will be the first failure again.
                # So we can just break the inner loop and let the outer loop handle it.
                # However, if we need 2 moves, and we only did 1, the outer loop will start again.
                # This is fine.
                fixed = True
                break
                
        if not fixed:
            # Stuck! No single prime change can fix x_fail without breaking the prefix.
            # This implies that for all p, changing r_p to cover x_fail breaks some x < x_fail.
            # This is a strong constraint.
            # Strategy: Random kick.
            # Pick a random prime and change its residue to a random value.
            # This might break the prefix, but might allow a different path.
            # To be safe, only do this if we've been stuck for a while?
            # Or, try to find a "swap" between two primes?
            
            # Let's try a "2-swap":
            # Change r_i and r_j simultaneously.
            # This is expensive to search.
            
            # Simpler: Random restart of the last few primes?
            # Or just a random move on a prime that covers x_fail?
            
            # Let's try: Pick a prime p that does NOT cover x_fail.
            # Change r_p to x_fail % p.
            # We already tried all of these and they failed the prefix check.
            
            # What if we pick a prime p that DOES cover x_fail?
            # Changing it will DECREASE count(x_fail). That's bad.
            
            # So we are truly stuck with 1-move.
            # Let's do a random perturbation:
            # Pick 2 random primes i, j.
            # Randomly change residues[i] and residues[j].
            # This is a "kick".
            
            i1 = np.random.randint(n_primes)
            i2 = np.random.randint(n_primes)
            if i1 == i2:
                i2 = (i2 + 1) % n_primes
            
            p1 = primes[i1]
            p2 = primes[i2]
            
            r1_old = residues[i1]
            r2_old = residues[i2]
            
            r1_new = random.randint(0, p1 - 1)
            r2_new = random.randint(0, p2 - 1)
            
            # Apply kick
            mask1_old = ((x_vals - r1_old) % p1 == 0).astype(np.int32)
            mask1_new = ((x_vals - r1_new) % p1 == 0).astype(np.int32)
            mask2_old = ((x_vals - r2_old) % p2 == 0).astype(np.int32)
            mask2_new = ((x_vals - r2_new) % p2 == 0).astype(np.int32)
            
            counts -= mask1_old
            counts += mask1_new
            counts -= mask2_old
            counts += mask2_new
            
            residues[i1] = r1_new
            residues[i2] = r2_new
            
            # This might make the score worse, but hopefully escapes the local minimum.
            # We keep best_residues, so we won't lose progress.
            
    return [int(r) for r in best_residues]
