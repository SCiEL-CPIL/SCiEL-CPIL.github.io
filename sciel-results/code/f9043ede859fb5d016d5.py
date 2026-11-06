import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute all arithmetic progressions as index lists (0-based)
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, a+2s, ..., a+(k-1)s where a+(k-1)s < n
    all_progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                all_progs.append(idx)
                
    Np_total = len(all_progs)
    
    # Strategy: Identify "critical" progressions.
    # Hypothesis: The maximum absolute sum is likely driven by:
    # 1. Long progressions (large k), because they sum more terms.
    # 2. Short progressions with small steps (dense), because they are numerous 
    #    and interact strongly.
    # 
    # We select a subset of progressions to constrain.
    # Let's select:
    # - All progressions with length >= 10 (long ones)
    # - All progressions with step s in {1, 2, 3, 4} (dense ones)
    # - Maybe some random others to cover edges?
    
    critical_ids = set()
    for pid, idx in enumerate(all_progs):
        k = len(idx)
        if k == 0: continue
        s = idx[1] - idx[0] if k > 1 else 1
        
        # Heuristic for criticality:
        # High length is critical.
        if k >= 8:
            critical_ids.add(pid)
        # Small steps are critical (more frequent interactions)
        if s <= 4:
            critical_ids.add(pid)
            
    # Ensure we have a reasonable number of constraints
    # If too few, add random ones. If too many, it's fine, but we want it small.
    # Let's check the count.
    # For n=256:
    # k>=8: Many.
    # s<=4: Many.
    
    # Actually, let's be more aggressive in reduction to speed up.
    # The "max absolute sum" is an L-infinity norm.
    # If we optimize the max over a subset S, the max over all P is >= max over S.
    # We want to minimize the max over all P.
    # If we only check S, we might miss a violation in P\S.
    # However, if S contains the "hardest" constraints, the optimum for S is likely close to optimum for P.
    # 
    # Let's refine the selection:
    # 1. All singletons (k=1) are trivial (sum is +/-1), ignore.
    # 2. All length 2 (k=2) are trivial (sum is 0, +2, -2). Max abs is 2. 
    #    Wait, if we have a run of +1, +1, sum is 2. 
    #    The global max will likely be > 2 if n=256.
    #    So short APs are rarely the binding constraint unless the solution is very bad.
    # 3. The binding constraints are likely medium-to-long APs.
    
    # New Selection Strategy:
    # Select progressions with length k in [10, 50] and step s in [1, 10].
    # Plus all progressions with s=1 (contiguous blocks) regardless of length? 
    # Contiguous blocks are very dense.
    
    critical_ids = set()
    for pid, idx in enumerate(all_progs):
        k = len(idx)
        if k < 5:
            continue
        s = idx[1] - idx[0] if k > 1 else 1
        
        # Include all contiguous (s=1)
        if s == 1:
            critical_ids.add(pid)
        # Include medium length with small step
        elif k >= 10 and s <= 5:
            critical_ids.add(pid)
        # Include long ones with any step
        elif k >= 20:
            critical_ids.add(pid)
            
    # Convert to list of indices
    critical_pids = list(critical_ids)
    Np_crit = len(critical_pids)
    
    # Build incidence matrix for critical progs only
    # M_crit[i, j] = 1 if position i is in critical prog j
    M_crit = np.zeros((n, Np_crit), dtype=np.float32)
    for j, pid in enumerate(critical_pids):
        idx = all_progs[pid]
        for i in idx:
            M_crit[i, j] = 1.0
            
    # Precompute row sums or other stats? Not needed for basic flip.
    
    def compute_sums(signs, M):
        return M.T @ signs
        
    def max_abs_sum(sums):
        if len(sums) == 0:
            return 0.0
        return np.max(np.abs(sums))
    
    # To verify the result against the FULL set of progs at the end,
    # we need the full incidence matrix or a way to compute full sums.
    # Computing full sums is O(n * Np_total) which is 256 * 32000 ~ 8M ops.
    # This is fast enough to do occasionally or at the end.
    
    M_full = np.zeros((n, Np_total), dtype=np.float32)
    for pid, idx in enumerate(all_progs):
        for i in idx:
            M_full[i, pid] = 1.0
            
    def evaluate_full(signs):
        sums_full = M_full.T @ signs
        return np.max(np.abs(sums_full))
        
    rng = np.random.RandomState(42)
    
    best_signs = None
    best_score_full = np.inf
    best_score_crit = np.inf
    
    # Multi-restart SA
    # We have more time per iteration now because M_crit is smaller.
    # Np_crit is likely a few thousand instead of 32k.
    # Let's estimate Np_crit.
    # s=1: ~250 progs.
    # k>=10, s<=5: 
    #   s=1: already counted.
    #   s=2: k>=10 means start <= 256-20=236. ~236 progs.
    #   s=3: k>=10 means start <= 256-30=226. ~226 progs.
    #   s=4: k>=10 means start <= 256-40=216. ~216 progs.
    #   s=5: k>=10 means start <= 256-50=206. ~206 progs.
    # k>=20, any s:
    #   s=6: k>=20 -> start <= 256-120=136. ~136.
    #   ...
    # Total Np_crit is probably around 2000-3000.
    # This is 10x faster per eval than full.
    
    n_restarts = 5
    iters_per_restart = 20000
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random init
        signs = rng.choice([-1, 1], size=n).astype(np.float32)
        sums_crit = compute_sums(signs, M_crit)
        score_crit = max_abs_sum(sums_crit)
        
        # Optional: Quick local search on full set to initialize? 
        # No, let's stick to crit set for speed.
        
        current_signs = signs.copy()
        current_sums = sums_crit.copy()
        current_score = score_crit
        
        T_start = 5.0
        T_end = 0.01
        
        for it in range(iters_per_restart):
            if time.time() > deadline - 5:
                break
                
            T = T_start * (1.0 - it / float(iters_per_restart)) + T_end
            
            i = rng.randint(n)
            sign_i = current_signs[i]
            
            # Flip i
            # delta = -2 * sign_i * M_crit[i, :]
            # new_sums = current_sums + delta
            # new_score = max(abs(new_sums))
            
            delta = -2.0 * sign_i * M_crit[i, :]
            new_sums = current_sums + delta
            new_score = np.max(np.abs(new_sums))
            
            delta_score = new_score - current_score
            
            if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
                current_signs[i] = -sign_i
                current_sums = new_sums
                current_score = new_score
                
                if current_score < best_score_crit:
                    best_score_crit = current_score
                    # We don't update best_signs_full here because crit score != full score
                    # But we can keep track of the candidate
                    candidate = current_signs.copy()
        
        # After each restart, evaluate the best candidate from this restart (or the global best)
        # Actually, we should evaluate the best candidate found so far against the FULL set.
        # Let's keep a global best candidate based on crit score, and periodically verify.
        
        # To save time, only verify if crit score improved significantly?
        # Or just verify the best crit candidate at the end of each restart.
        
        if best_signs is None or best_score_crit < best_score_full - 0.5: # Heuristic: if crit is much better, it might be full better too
            # Evaluate the current best candidate (which is current_signs if it was the best in this restart? 
            # No, we didn't track the best candidate of the restart separately from the global best_crit.
            # Let's just evaluate the current_signs at the end of the restart.
            pass
            
    # The logic above is a bit messy regarding which candidate to evaluate.
    # Let's restructure:
    # 1. Run SA on crit set.
    # 2. Keep the best signs found in terms of crit score.
    # 3. At the very end, evaluate that best signs on full set.
    # 4. If time permits, do a few full-set local search steps to polish.
    
    # Let's redo the loop structure for clarity and correctness.
    
    best_signs_crit = None
    best_val_crit = np.inf
    
    n_restarts = 10
    iters_per_restart = 10000
    
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
            
        signs = rng.choice([-1, 1], size=n).astype(np.float32)
        sums_crit = compute_sums(signs, M_crit)
        score_crit = max_abs_sum(sums_crit)
        
        current_signs = signs.copy()
        current_sums = sums_crit.copy()
        current_score = score_crit
        
        T_start = 5.0
        T_end = 0.01
        
        for it in range(iters_per_restart):
            if time.time() > deadline - 10:
                break
                
            T = T_start * (1.0 - it / float(iters_per_restart)) + T_end
            
            i = rng.randint(n)
            sign_i = current_signs[i]
            
            delta = -2.0 * sign_i * M_crit[i, :]
            new_sums = current_sums + delta
            new_score = np.max(np.abs(new_sums))
            
            delta_score = new_score - current_score
            
            if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
                current_signs[i] = -sign_i
                current_sums = new_sums
                current_score = new_score
                
                if current_score < best_val_crit:
                    best_val_crit = current_score
                    best_signs_crit = current_signs.copy()
                    
    # Phase 2: Polish the best candidate using FULL constraints
    # Take best_signs_crit and run a short local search minimizing FULL max abs sum.
    
    if best_signs_crit is not None:
        signs = best_signs_crit.copy()
    else:
        signs = rng.choice([-1, 1], size=n).astype(np.float32)
        
    sums_full = compute_sums(signs, M_full)
    score_full = max_abs_sum(sums_full)
    
    best_full_signs = signs.copy()
    best_full_score = score_full
    
    # Short SA on full set
    # Np_full is ~32k. 
    # We have ~10-15 seconds left.
    # Each full eval is ~10x slower than crit.
    # So we can do ~1/10th the iterations.
    # Let's do 5000 iterations.
    
    n_iters_full = 5000
    T_start = 2.0
    T_end = 0.01
    
    for it in range(n_iters_full):
        if time.time() > deadline - 2:
            break
            
        T = T_start * (1.0 - it / float(n_iters_full)) + T_end
        
        i = rng.randint(n)
        sign_i = signs[i]
        
        delta = -2.0 * sign_i * M_full[i, :]
        new_sums = sums_full + delta
        new_score = np.max(np.abs(new_sums))
        
        delta_score = new_score - score_full
        
        if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
            signs[i] = -sign_i
            sums_full = new_sums
            score_full = new_score
            
            if score_full < best_full_score:
                best_full_score = score_full
                best_full_signs = signs.copy()
                
    return [int(x) for x in best_full_signs]
