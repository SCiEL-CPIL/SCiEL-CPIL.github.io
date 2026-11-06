import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Precompute APs
    # We categorize APs into "Hard" (likely to be the bottleneck) and "Soft" (for final validation)
    # Hard APs: Long lengths, small steps. These have many terms and are harder to cancel out.
    # Specifically, APs with length >= 10 and step <= 10 are often the worst offenders.
    # Also include all length-2 APs (step 1) and length-3 APs (step 1) as they are dense.
    
    all_aps = []
    hard_aps = []
    soft_aps = []
    
    for a in range(n):
        for s in range(1, n):
            # Generate indices
            idx = list(range(a, n, s))
            k = len(idx)
            if k < 1:
                continue
            
            # Heuristic for "Hardness":
            # 1. Small step s (dense in the array)
            # 2. Large length k (more terms to sum)
            # 3. Specifically, s=1 is the most critical (intervals)
            
            is_hard = False
            if s == 1 and k >= 5:
                is_hard = True
            elif s <= 5 and k >= 8:
                is_hard = True
            elif k >= 15: # Very long APs regardless of step (though step will be large)
                is_hard = True
            
            if is_hard:
                hard_aps.append(idx)
            else:
                soft_aps.append(idx)
                
    # Ensure we have a decent number of hard APs. If not, add some random ones from soft?
    # Actually, let's just use the hard list for optimization.
    # We need to be careful: optimizing only on hard APs might lead to bad soft APs.
    # Strategy: Optimize on hard APs, but if we get stuck or improve, check soft APs.
    # To make the local search efficient, we will maintain sums for hard APs.
    
    N_hard = len(hard_aps)
    N_soft = len(soft_aps)
    
    # Build incidence matrices
    # M_hard: (N_hard, n)
    M_hard = np.zeros((N_hard, n), dtype=np.int8)
    for pid, idx in enumerate(hard_aps):
        for i in idx:
            M_hard[pid, i] = 1
            
    M_soft = np.zeros((N_soft, n), dtype=np.int8)
    for pid, idx in enumerate(soft_aps):
        for i in idx:
            M_soft[pid, i] = 1

    # Precompute column norms or counts?
    # For flip evaluation:
    # new_sum = old_sum - 2 * sign * M_col
    
    rng = np.random.RandomState(42)
    
    best_signs = None
    best_score = n # Upper bound
    
    def get_full_score(signs):
        """Compute max abs sum over ALL APs."""
        sums_hard = M_hard @ signs
        sums_soft = M_soft @ signs
        return int(max(np.max(np.abs(sums_hard)), np.max(np.abs(sums_soft))))
    
    def get_hard_score(signs):
        """Compute max abs sum over HARD APs only (faster)."""
        sums_hard = M_hard @ signs
        return int(np.max(np.abs(sums_hard)))
    
    def local_search(signs, max_time=5.0):
        """
        Perform local search optimizing primarily for Hard APs,
        but occasionally checking Full Score to avoid drifting into bad Soft APs.
        Uses single flips and pair flips.
        """
        current_signs = signs.copy()
        current_sums_hard = M_hard @ current_signs
        current_hard_score = int(np.max(np.abs(current_sums_hard)))
        
        # Initial full check
        current_full_score = get_full_score(current_signs)
        
        local_best_signs = current_signs.copy()
        local_best_full_score = current_full_score
        local_best_hard_score = current_hard_score
        
        last_improvement_time = time.time()
        
        while time.time() < max_time:
            # Evaluate all single flips on Hard APs
            # changes = 2 * signs[None, :] * M_hard
            # new_sums = current_sums_hard[:, None] - changes
            # new_hard_scores = np.max(np.abs(new_sums), axis=0)
            
            # Optimization: We only care about flips that reduce the max hard score
            # or keep it same but might improve full score.
            
            changes = 2.0 * current_signs[None, :] * M_hard.astype(np.float32)
            new_sums = current_sums_hard[:, None] - changes
            new_hard_scores = np.max(np.abs(new_sums), axis=0)
            
            # Find candidates that improve or maintain hard score
            # If hard score improves, it's a strong candidate.
            # If hard score stays same, we might check full score for a few best candidates.
            
            # Sort indices by new_hard_scores
            # We want the smallest new_hard_scores.
            
            # To save time, just pick the best few.
            # If the best new_hard_score < current_hard_score, take it.
            # Else, if best == current, check if it helps full score.
            
            best_idx = np.argmin(new_hard_scores)
            best_new_hard = new_hard_scores[best_idx]
            
            accepted = False
            
            if best_new_hard < current_hard_score:
                # Always accept if hard score improves
                # Apply flip
                current_signs[best_idx] = -current_signs[best_idx]
                current_sums_hard -= 2.0 * (-current_signs[best_idx]) * M_hard[:, best_idx] # Wait, sign flipped.
                # Let's be precise:
                # old_sign = -current_signs[best_idx]
                # new_sign = current_signs[best_idx]
                # change = new_sign - old_sign = 2 * new_sign
                # sum += change * M_col
                # So: current_sums_hard += 2.0 * current_signs[best_idx] * M_hard[:, best_idx]
                
                # Correction:
                # I flipped current_signs[best_idx] above.
                # old_sign was -current_signs[best_idx] (current is new).
                # So update: current_sums_hard += 2.0 * current_signs[best_idx] * M_hard[:, best_idx]
                
                # Wait, I already updated current_signs.
                # Let's redo the update logic cleanly.
                
                # Re-evaluate sums from scratch is O(N_hard * n).
                # Incremental is O(N_hard).
                # Let's use incremental.
                
                # State before flip:
                # sign_old = -current_signs[best_idx]
                # sum_old = current_sums_hard
                # State after flip:
                # sign_new = current_signs[best_idx]
                # sum_new = sum_old + (sign_new - sign_old) * M[:, idx]
                # sum_new = sum_old + 2 * sign_new * M[:, idx]
                
                current_sums_hard += 2.0 * current_signs[best_idx] * M_hard[:, best_idx].astype(np.float32)
                current_hard_score = int(best_new_hard)
                accepted = True
                
            elif best_new_hard == current_hard_score:
                # Check if this flip improves Full Score
                # Only do this for a few top candidates to save time
                candidates = np.argsort(new_hard_scores)[:5]
                found_improvement = False
                for idx in candidates:
                    if new_hard_scores[idx] > current_hard_score:
                        break # Since sorted, rest are worse
                    # Test flip at idx
                    # Temporarily flip
                    old_sign = current_signs[idx]
                    current_signs[idx] = -old_sign
                    # Update sums
                    current_sums_hard += 2.0 * (-old_sign) * M_hard[:, idx].astype(np.float32)
                    
                    # Check full score
                    full_score = get_full_score(current_signs)
                    
                    if full_score < local_best_full_score:
                        # Accept this flip permanently
                        current_hard_score = int(new_hard_scores[idx])
                        local_best_full_score = full_score
                        local_best_signs = current_signs.copy()
                        last_improvement_time = time.time()
                        accepted = True
                        found_improvement = True
                        break
                    else:
                        # Revert
                        current_signs[idx] = old_sign
                        current_sums_hard -= 2.0 * (-old_sign) * M_hard[:, idx].astype(np.float32)
                
                if found_improvement:
                    continue
            
            if accepted:
                # Check if full score improved for the accepted hard-improving move
                full_score = get_full_score(current_signs)
                if full_score < local_best_full_score:
                    local_best_full_score = full_score
                    local_best_signs = current_signs.copy()
                    last_improvement_time = time.time()
                elif full_score > local_best_full_score + 2:
                    # We drifted too far into bad full scores?
                    # Maybe revert to local_best?
                    # For now, just continue, hoping to find a path back.
                    pass
            
            # If no improvement in hard score for a while, try random kick?
            # Or just break if stuck.
            if time.time() - last_improvement_time > 2.0:
                # Stagnation in local search
                # Try a random kick: flip 2 random positions
                if rng.random() < 0.5:
                    idx1, idx2 = rng.choice(n, size=2, replace=False)
                    current_signs[idx1] = -current_signs[idx1]
                    current_signs[idx2] = -current_signs[idx2]
                    current_sums_hard = M_hard @ current_signs
                    current_hard_score = int(np.max(np.abs(current_sums_hard)))
                    last_improvement_time = time.time()
                    
        return local_best_signs, local_best_full_score, local_best_hard_score

    # Multi-Start
    # Start 1: Random
    # Start 2: Thue-Morse
    # Start 3: Alternating
    
    starts = []
    
    # Random
    starts.append(rng.choice([-1, 1], size=n).astype(np.float32))
    
    # Thue-Morse
    tm = np.zeros(n, dtype=np.float32)
    for i in range(n):
        # Count set bits
        # i & (i-1) ...
        # Simpler:
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        tm[i] = 1 if c % 2 == 0 else -1
    starts.append(tm)
    
    # Alternating
    alt = np.ones(n, dtype=np.float32)
    alt[1::2] = -1
    starts.append(alt)
    
    # Maybe a "Wave" pattern?
    wave = np.sin(np.linspace(0, 10*np.pi, n))
    wave[wave >= 0] = 1
    wave[wave < 0] = -1
    starts.append(wave.astype(np.float32))

    for i, init_signs in enumerate(starts):
        if time.time() > deadline - 5:
            break
            
        # Run local search with a time slice
        # Give more time to later starts? Or equal?
        # Let's give 10s per start, up to 4 starts = 40s.
        # Plus overhead.
        
        time_slice = 10.0
        if i == 0:
            time_slice = 15.0 # First start gets more time
        
        best_signs_local, best_full, best_hard = local_search(init_signs, max_time=time_slice)
        
        if best_full < best_score:
            best_score = best_full
            best_signs = best_signs_local.copy()
            
        # If we are very good, stop early?
        if best_score <= 2:
            break

    # Final Polish: Greedy single flip on the best solution found
    if time.time() < deadline - 1 and best_signs is not None:
        signs = best_signs.astype(np.float32)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Evaluate all flips
            current_sums_hard = M_hard @ signs
            current_sums_soft = M_soft @ signs
            current_full = int(max(np.max(np.abs(current_sums_hard)), np.max(np.abs(current_sums_soft))))
            
            # We want to find a flip that reduces current_full.
            # This requires checking full sums for each flip.
            # This is O(n * (N_hard + N_soft)).
            # N_hard ~ 2000, N_soft ~ 30000. Total ~ 32000.
            # 256 * 32000 = 8M ops. Fast in numpy.
            
            changes_hard = 2.0 * signs[None, :] * M_hard.astype(np.float32)
            new_sums_hard = current_sums_hard[:, None] - changes_hard
            
            changes_soft = 2.0 * signs[None, :] * M_soft.astype(np.float32)
            new_sums_soft = current_sums_soft[:, None] - changes_soft
            
            new_maxes_hard = np.max(np.abs(new_sums_hard), axis=0)
            new_maxes_soft = np.max(np.abs(new_sums_soft), axis=0)
            
            new_maxes_full = np.maximum(new_maxes_hard, new_maxes_soft)
            
            best_idx = np.argmin(new_maxes_full)
            best_new_score = new_maxes_full[best_idx]
            
            if best_new_score < current_full:
                signs[best_idx] = -signs[best_idx]
                improved = True
                best_score = int(best_new_score)
                best_signs = signs.copy()
            else:
                break

    return [int(x) for x in best_signs]
