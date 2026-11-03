import time
import random
import numpy as np

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0  # Safety margin
    total = q ** n
    rng = random.Random(42)
    
    # Precompute balls
    # ball_masks[i] is a boolean array of length total, True if j is in ball(i)
    # For q=3, n=6, radius=1, ball size is 13.
    # We can store these as bitsets or just boolean arrays.
    # Given total=729, boolean arrays are 729 bytes each. 729*729 ~ 500KB. Fine.
    
    ball_masks = np.zeros((total, total), dtype=bool)
    # Generate neighbors
    for i in range(total):
        ball_masks[i, i] = True
        for pos in range(n):
            digit = (i // (q ** pos)) % q
            for new_digit in range(q):
                if new_digit != digit:
                    j = i + (new_digit - digit) * (q ** pos)
                    ball_masks[i, j] = True

    # Greedy initialization
    def greedy_init():
        counts = np.zeros(total, dtype=np.int32)
        code = []
        remaining = np.ones(total, dtype=bool)
        
        while remaining.any():
            if time.monotonic() > deadline:
                break
            # Gain for each c is number of remaining points in ball(c)
            gains = ball_masks @ remaining
            best_c = np.argmax(gains)
            if gains[best_c] == 0:
                break
            code.append(best_c)
            counts += ball_masks[best_c]
            remaining = (counts == 0)
        
        # Fix any remaining (should be none if loop finished)
        while remaining.any():
            # Pick any c that covers a remaining point
            # For efficiency, just pick random until covered
            rem_indices = np.where(remaining)[0]
            target = rem_indices[0]
            # Find a c covering target
            candidates = np.where(ball_masks[:, target])[0]
            c = candidates[0]
            if c not in code:
                code.append(c)
                counts += ball_masks[c]
                remaining = (counts == 0)
        
        return code, counts

    def remove_redundant(code, counts):
        code_set = set(code)
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if any point in ball(c) has count == 1
                if np.all(counts[ball_masks[c]] > 1):
                    code.pop(i)
                    code_set.discard(c)
                    counts -= ball_masks[c]
                    changed = True
        return code, counts

    def perturb_repair(code, counts, budget):
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        
        # We will iterate: Perturb -> Repair -> Remove Redundant
        # Perturb: Remove k random codewords
        # Repair: Add codewords to cover uncovered points (Greedy)
        
        while time.monotonic() < budget:
            # 1. Remove redundant
            code, counts = remove_redundant(code, counts)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                
            if len(code) <= 73: # We found a very good solution, maybe try to go lower?
                # Actually, 73 is the target. If we hit 73, we can stop or try for 72.
                # But 72 is likely impossible.
                pass

            # 2. Perturb
            # Remove 1 to 3 random codewords
            k = rng.randint(1, 3)
            if len(code) < k:
                continue
            
            indices_to_remove = rng.sample(range(len(code)), k)
            # Sort to remove from end to avoid index shifting issues, or just collect and remove
            # Easier: mark them, then filter
            remove_set = set([code[i] for i in indices_to_remove])
            new_code = [c for c in code if c not in remove_set]
            new_counts = np.zeros(total, dtype=np.int32)
            for c in new_code:
                new_counts += ball_masks[c]
            
            # 3. Repair
            # While uncovered points exist:
            #   Find c that covers max uncovered points
            #   Add c
            uncovered = (new_counts == 0)
            while uncovered.any():
                if time.monotonic() > budget:
                    break
                # Gains: number of uncovered points covered by each c
                # We only care about c not in new_code
                # For speed, we can just compute gains for all, and ignore those in code
                gains = ball_masks @ uncovered
                # Mask out current code
                for c in new_code:
                    gains[c] = -1
                
                best_c = np.argmax(gains)
                if gains[best_c] <= 0:
                    # Should not happen if uncovered.any()
                    break
                
                new_code.append(best_c)
                new_counts += ball_masks[best_c]
                uncovered = (new_counts == 0)
            
            # 4. Update state
            code = new_code
            counts = new_counts
            
        return best_code, best_size

    # Main Loop
    # Start with greedy
    code, counts = greedy_init()
    code, counts = remove_redundant(code, counts)
    best_code = code[:]
    best_size = len(code)
    
    # If greedy is already good, we can try to improve
    # Otherwise, run perturb-repair
    
    # We can run multiple restarts if time permits, but perturb-repair on one good init is often enough.
    # Let's do a few restarts of the whole process if time allows.
    
    # Strategy: 
    # 1. Greedy
    # 2. Remove Redundant
    # 3. Perturb-Repair loop for remaining time
    
    # To be safe, let's try a few different greedy seeds or perturbations.
    
    # Actually, the perturb-repair is a local search around the greedy solution.
    # If greedy is bad, we might need a better start.
    
    # Let's try 3 different greedy starts (with random tie-breaking)
    
    def greedy_init_random(seed):
        r = random.Random(seed)
        counts = np.zeros(total, dtype=np.int32)
        code = []
        remaining = np.ones(total, dtype=bool)
        
        while remaining.any():
            if time.monotonic() > deadline:
                break
            gains = ball_masks @ remaining
            best_gain = gains.max()
            if best_gain == 0:
                break
            candidates = np.where(gains == best_gain)[0]
            # Random pick from candidates
            best_c = candidates[r.randint(0, len(candidates)-1)]
            code.append(best_c)
            counts += ball_masks[best_c]
            remaining = (counts == 0)
            
        # Fix
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            target = rem_indices[0]
            candidates = np.where(ball_masks[:, target])[0]
            c = candidates[0]
            if c not in code:
                code.append(c)
                counts += ball_masks[c]
                remaining = (counts == 0)
                
        return code, counts

    # Try 3 restarts
    for seed in [42, 123, 456]:
        if time.monotonic() > deadline - 10:
            break
            
        code, counts = greedy_init_random(seed)
        code, counts = remove_redundant(code, counts)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
        
        # Run perturb-repair on this solution
        # Allocate remaining time / 3 for this restart
        time_left = deadline - time.monotonic()
        if time_left < 5:
            continue
            
        # We want to spend most of the time on the best solution so far?
        # Or spend equal time on each?
        # Let's spend equal time on each restart.
        # But if one restart finds 73, we should focus on it?
        # Actually, if we find 73, we are done (best known).
        # If we find 74, we try to improve.
        
        # Let's just run perturb-repair for a fixed fraction of remaining time.
        # Simpler: Run perturb-repair until deadline, but keep track of best.
        # But we have 3 restarts.
        
        # Let's do this:
        # Restart 1: Greedy + Polish (1/3 time)
        # Restart 2: Greedy + Polish (1/3 time)
        # Restart 3: Greedy + Polish (1/3 time)
        
        # This is what I will implement.
        
        # Calculate budget for this restart
        # We are in a loop, so we need to manage time carefully.
        # Let's just run perturb_repair with a budget of (deadline - time.monotonic()) / (3 - seed_index)?
        # No, simpler:
        
        # I will restructure to run 3 distinct phases.
        
    # Restructured:
    
    # Phase 1: Find a good initial solution
    # Phase 2: Optimize it
    
    # Let's just run the perturb-repair on the best solution found so far,
    # but allow it to "restart" internally if it gets stuck?
    # No, the perturb step is the restart.
    
    # Let's run 3 independent "Greedy + Perturb-Repair" cycles.
    
    best_code = None
    best_size = total + 1
    
    num_cycles = 3
    cycle_time = (deadline - time.monotonic()) / num_cycles
    
    for i in range(num_cycles):
        if time.monotonic() > deadline - 2:
            break
            
        # 1. Greedy
        code, counts = greedy_init_random(42 + i*100)
        code, counts = remove_redundant(code, counts)
        
        # 2. Perturb-Repair
        # Budget is cycle_time - time_spent_on_greedy
        # Greedy is fast, so just use cycle_time
        polished_code, polished_size = perturb_repair(code, counts, time.monotonic() + cycle_time)
        
        if polished_size < best_size:
            best_size = polished_size
            best_code = polished_code[:]
            
    if best_code is None:
        # Fallback
        code, counts = greedy_init()
        best_code = code
        
    # Convert to words
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
