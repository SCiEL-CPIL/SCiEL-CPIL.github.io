import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We will evaluate up to a sufficiently large L to ensure we can find improvements
    # The score is the largest L such that 1..L is doubly covered.
    # We cap the search space for evaluation at MAX_EVAL to keep memory/time low.
    MAX_EVAL = 20000 
    
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def evaluate(residues):
        """
        Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
        Returns L (capped at MAX_EVAL).
        """
        # coverage[x] = number of primes p_i such that x % p_i == r_i
        # We compute this vectorized.
        # diff[i, j] = xs[j] - residues[i]
        # mask[i, j] = (diff[i, j] % primes[i]) == 0
        # coverage = sum(mask, axis=0)
        
        # To save memory, we can do this in chunks or just rely on numpy efficiency for 168x20000
        # 168 * 20000 = 3.36 million elements. Boolean is 1 byte. ~3.3MB. Very fast.
        diff = xs[None, :] - residues[:, None]
        # Modulo operation
        # Note: residues and primes are int64. 
        # We need to be careful with negative numbers in modulo, but numpy handles % correctly 
        # (result has sign of divisor, which is positive primes, so result is 0..p-1).
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        
        # Find the first index where coverage < 2
        # If all are >= 2, return MAX_EVAL
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        
        # argmax returns the index of the first True
        # xs is 1-indexed in value (1..MAX_EVAL), but array index 0 corresponds to x=1.
        # So if mask[0] is True, L=0? No, the score is "largest L such that 1..L satisfies".
        # If x=1 fails, L=0.
        # If x=1,2 pass, x=3 fails, L=2.
        # mask index 0 corresponds to x=1.
        # If mask[0] is False, mask[1] is True -> L=1.
        # So L = index_of_first_fail - 1? 
        # Wait. If mask[k] is True, then x = k+1 is the first failure.
        # So all x < k+1 are good. The largest L is k.
        first_fail_idx = np.argmax(mask)
        return int(first_fail_idx)

    # Initial solution: 
    # A good heuristic start is to try to cover small numbers densely.
    # For each prime p, choose r_p such that it covers a small number.
    # However, we need *double* coverage.
    # Let's start with a randomized start that is biased towards small residues.
    # Or use the "seed" logic if it exists, but let's generate a smart random start.
    
    def generate_initial_residues():
        res = np.zeros(n_primes, dtype=np.int64)
        for i, p in enumerate(primes):
            # Bias towards small residues to help cover small x's
            # r_p = x % p for some small x?
            # Let's just pick random but ensure we don't start with all 0s which might be bad
            # Actually, setting r_p = 0 for all p means x=0 is covered by all, but x=1 is covered by none (if p>1).
            # We need x=1 to be covered by at least 2 primes.
            # If r_p = 1, then x=1 is covered.
            # Let's try a mix.
            res[i] = random.randint(0, p - 1)
        return res

    # Strategy: 
    # 1. Start with a reasonable random initialization.
    # 2. Use a "Greedy Repair" loop:
    #    - Find current L.
    #    - Identify the "critical" integer x = L+1 which has coverage < 2.
    #    - Identify primes that currently cover x (coverage contribution 1) or could cover x (if we change residue).
    #    - Specifically, find primes p_i such that changing r_i to (L+1) % p_i would increase coverage of x.
    #    - Among these, pick one and change its residue.
    #    - Check if this breaks coverage of previous integers (1..L). 
    #    - If it does, we might need to revert or fix another prime.
    #    - This is complex. 
    #    
    # Alternative Strategy: Local Search with "Boundary Focus"
    # Instead of random moves, we focus moves on primes that affect the boundary L.
    # 
    # Let's combine Simulated Annealing with a smart move generator:
    # 1. Evaluate current L.
    # 2. If L < MAX_EVAL, look at x = L+1.
    # 3. Count coverage of x. Let c_x be the count.
    # 4. We need c_x >= 2.
    # 5. Candidates for improvement:
    #    a. Primes p_i that DO NOT cover x. If we change r_i to x % p_i, c_x increases by 1.
    #    b. Primes p_i that DO cover x. Changing r_i might keep c_x same or decrease it.
    #    
    # So, a promising move is to pick a prime p_i that does NOT cover x, and set r_i = x % p_i.
    # However, this might break coverage of some y in 1..L.
    # 
    # Let's implement a hybrid:
    # - Most of the time, try "Repair Moves": pick a prime not covering L+1, set it to cover L+1.
    # - Sometimes, try "Random Moves" to escape local minima.
    # - Use SA to accept/reject.

    # Initial residues
    residues = generate_initial_residues()
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    # Precompute coverage array for current state to speed up delta calculation?
    # Full evaluation is ~3.3MB ops. 120s / 1ms = 120k evals possible.
    # Let's just use full evaluation for simplicity and correctness, it's fast enough.
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Cache coverage for current state to help with move generation
    def get_coverage(residues):
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    current_coverage = get_coverage(residues)
    
    while time.time() < deadline:
        # Determine move type
        if current_score < MAX_EVAL:
            x_target = current_score + 1  # The first integer that is not doubly covered
            # Check coverage of x_target
            # Note: xs is 1-indexed in values. Index is x_target - 1.
            idx_target = x_target - 1
            c_target = current_coverage[idx_target]
            
            # Find primes that do NOT cover x_target
            # A prime i covers x_target if (x_target - residues[i]) % primes[i] == 0
            # We want to change residues[i] to x_target % primes[i]
            
            # Identify candidate indices
            # For all i, check if they cover x_target
            # diff = x_target - residues
            # mods = diff % primes
            # covers = (mods == 0)
            
            diff_target = x_target - residues
            mods_target = np.remainder(diff_target, primes_arr)
            covers_target = (mods_target == 0)
            
            # Candidates for "Repair": indices where covers_target is False
            candidate_indices = np.where(~covers_target)[0]
            
            if len(candidate_indices) > 0:
                # Pick a random candidate
                # Prefer smaller primes? Smaller primes have higher frequency, so changing them
                # is more likely to break other coverages? Or maybe less?
                # Actually, changing a large prime affects fewer numbers in 1..L.
                # Let's pick a random one.
                move_type = "repair"
                i = candidate_indices[random.randint(0, len(candidate_indices) - 1)]
                new_r = x_target % primes_arr[i]
                
                # Apply move
                old_r = residues[i]
                residues[i] = new_r
                
                # Evaluate
                new_score = evaluate(residues)
                delta = new_score - current_score
                
                # Update coverage for next iteration? 
                # We can update current_coverage incrementally or recompute.
                # Recomputing is safer.
                current_coverage = get_coverage(residues)
                
                if delta > 0:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                else:
                    # Accept with prob exp(delta/T)
                    if delta == 0:
                        accept_prob = 0.5
                    else:
                        try:
                            accept_prob = np.exp(delta / T)
                        except OverflowError:
                            accept_prob = 0.0
                            
                    if random.random() < accept_prob:
                        current_score = new_score
                    else:
                        # Revert
                        residues[i] = old_r
                        # Revert coverage? We already recomputed it for the new state.
                        # If we revert, we need to revert coverage too.
                        # This is inefficient. 
                        # Let's just recompute coverage after the decision.
                        pass 
            else:
                # All primes cover x_target. We need to change one to *not* cover it? No, we need MORE coverage.
                # Wait, if all primes cover x_target, then c_target = 168. That's >= 2.
                # So this case (c_target < 2) implies NOT all primes cover it.
                # So candidate_indices should never be empty if c_target < 2.
                # Unless my logic is wrong.
                # If c_target < 2, then at most 1 prime covers it.
                # So at least 167 primes do NOT cover it.
                # So candidate_indices is large.
                
                # This branch should be rare/impossible if c_target < 2.
                # If c_target >= 2, we wouldn't be targeting x_target for repair.
                # But we only enter this block if current_score < MAX_EVAL, so x_target is the first failure.
                # So c_target MUST be < 2.
                
                # Fallback: Random move
                move_type = "random"
                i = random.randint(0, n_primes - 1)
                new_r = random.randint(0, primes_arr[i] - 1)
                old_r = residues[i]
                residues[i] = new_r
                
                new_score = evaluate(residues)
                delta = new_score - current_score
                current_coverage = get_coverage(residues)
                
                if delta > 0:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                else:
                    if delta == 0:
                        accept_prob = 0.5
                    else:
                        try:
                            accept_prob = np.exp(delta / T)
                        except OverflowError:
                            accept_prob = 0.0
                            
                    if random.random() < accept_prob:
                        current_score = new_score
                    else:
                        residues[i] = old_r
                        # Recompute coverage for reverted state
                        current_coverage = get_coverage(residues)
        else:
            # We reached MAX_EVAL. Try to push further? 
            # The task says capped at 10^5. MAX_EVAL is 20000.
            # If we hit 20000, we can try to increase MAX_EVAL? 
            # Or just stop. 20000 is already > 1000.
            # Let's just do random moves to see if we can find a structure that allows higher L.
            # But evaluate is capped at MAX_EVAL. 
            # To go higher, we'd need to increase MAX_EVAL and re-evaluate.
            # For now, if we hit cap, we just do random moves to maintain state.
            i = random.randint(0, n_primes - 1)
            new_r = random.randint(0, primes_arr[i] - 1)
            old_r = residues[i]
            residues[i] = new_r
            
            new_score = evaluate(residues)
            delta = new_score - current_score
            current_coverage = get_coverage(residues)
            
            if delta > 0:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues.copy()
            else:
                if delta == 0:
                    accept_prob = 0.5
                else:
                    try:
                        accept_prob = np.exp(delta / T)
                    except OverflowError:
                        accept_prob = 0.0
                        
                if random.random() < accept_prob:
                    current_score = new_score
                else:
                    residues[i] = old_r
                    current_coverage = get_coverage(residues)

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

        # Periodic restart if stuck?
        # If no improvement in last 1000 steps, reheat?
        # Simple version: just keep going.

        if time.time() > deadline - 5.0:
            break

    return best_residues.tolist()
