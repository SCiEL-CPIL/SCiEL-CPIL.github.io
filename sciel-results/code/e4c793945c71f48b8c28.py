import time
import random
import numpy as np

_BASE = 51975
_DEADLINE = 115.0

def _get_divisors(n):
    divs = set()
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    divs.add(n)
    return sorted(divs)

_DIVS = _get_divisors(_BASE)
_DIVS_SET = set(_DIVS)

def construct():
    start_time = time.time()
    time_limit = _DEADLINE
    
    N = _BASE
    divisors = _DIVS
    n_divs = len(divisors)
    div_idx = {m: i for i, m in enumerate(divisors)}
    
    # Precompute indices for each modulus and residue
    indices_cache = {}
    for m in divisors:
        indices_cache[m] = [np.arange(r, N, m) for r in range(m)]
    
    def compute_uncovered(residues):
        """Compute uncovered count from a residues dict."""
        counts = np.zeros(N, dtype=np.int8)
        for m in divisors:
            counts[indices_cache[m][residues[m]]] += 1
        return int(np.sum(counts == 0))
    
    def greedy_build(order):
        """Greedy: process moduli in given order, pick best residue."""
        covered = np.zeros(N, dtype=bool)
        residues = {}
        for m in order:
            best_r = 0
            best_gain = -1
            for r in range(m):
                idx = indices_cache[m][r]
                gain = int((~covered[idx]).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[m] = best_r
            covered[indices_cache[m][best_r]] = True
        return residues

    # 1. Initialization: Try multiple greedy orders
    rng = random.Random(42)
    
    # Order 1: Ascending
    res_asc = greedy_build(divisors)
    unc_asc = compute_uncovered(res_asc)
    
    # Order 2: Descending
    res_desc = greedy_build(sorted(divisors, reverse=True))
    unc_desc = compute_uncovered(res_desc)
    
    # Order 3-5: Random
    best_res = res_asc if unc_asc <= unc_desc else res_desc
    best_unc = min(unc_asc, unc_desc)
    
    for _ in range(5):
        order = divisors.copy()
        rng.shuffle(order)
        res_rand = greedy_build(order)
        unc_rand = compute_uncovered(res_rand)
        if unc_rand < best_unc:
            best_unc = unc_rand
            best_res = res_rand

    best_residues = best_res.copy()
    best_uncovered = best_unc
    
    # 2. Local Search with precomputed count sums
    # counts[x] = number of residue classes covering position x
    # cnt1[m][r] = number of positions in class (m, r) where counts == 1
    # cnt0[m][r] = number of positions in class (m, r) where counts == 0
    
    counts = np.zeros(N, dtype=np.int8)
    for m in divisors:
        counts[indices_cache[m][best_residues[m]]] += 1
    
    # Precompute cnt1 and cnt0 for all (m, r)
    cnt1 = {}
    cnt0 = {}
    for m in divisors:
        cnt1[m] = np.zeros(m, dtype=np.int32)
        cnt0[m] = np.zeros(m, dtype=np.int32)
        for r in range(m):
            idx = indices_cache[m][r]
            cnt1[m][r] = np.sum(counts[idx] == 1)
            cnt0[m][r] = np.sum(counts[idx] == 0)
    
    current_uncovered = best_uncovered
    
    def update_cnt(m, r_old, r_new):
        """Update cnt1 and cnt0 after moving m from r_old to r_new."""
        nonlocal counts
        idx_old = indices_cache[m][r_old]
        idx_new = indices_cache[m][r_new]
        
        # Remove r_old
        counts[idx_old] -= 1
        # Add r_new
        counts[idx_new] += 1
        
        # Recompute cnt1[m] and cnt0[m]
        for r in range(m):
            idx = indices_cache[m][r]
            cnt1[m][r] = np.sum(counts[idx] == 1)
            cnt0[m][r] = np.sum(counts[idx] == 0)
    
    no_improve_steps = 0
    max_no_improve = 300
    step_count = 0
    
    while time.time() - start_time < time_limit:
        step_count += 1
        # Pick a random modulus to move
        m = rng.choice(divisors)
        r_old = best_residues[m]
        
        best_r = r_old
        best_delta = 0
        
        # L = cnt1[m][r_old]
        L = int(cnt1[m][r_old])
        
        # G = cnt0[m][r_new] for each r_new
        # delta = L - G
        # We want to minimize delta, so maximize G
        # But we need delta < 0, i.e., G > L
        
        if L > 0:
            # Find r_new with max cnt0[m][r_new]
            max_G = np.max(cnt0[m])
            if max_G > L:
                # Find all r_new with cnt0[m][r_new] == max_G
                candidates = np.where(cnt0[m] == max_G)[0]
                # Exclude r_old if it's in candidates
                candidates = candidates[candidates != r_old]
                if len(candidates) > 0:
                    # Pick random from best candidates
                    best_r = int(rng.choice(candidates))
                    best_delta = L - int(max_G)
        
        if best_r != r_old:
            # Apply move
            update_cnt(m, r_old, best_r)
            
            current_uncovered += best_delta
            best_residues[m] = best_r
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        # Escape local minimum if stuck
        if no_improve_steps > max_no_improve:
            # Pick a random m and random r
            m_rand = rng.choice(divisors)
            r_rand = rng.randint(0, m_rand - 1)
            
            r_old_rand = best_residues[m_rand]
            if r_rand != r_old_rand:
                # Compute delta
                L_rand = int(cnt1[m_rand][r_old_rand])
                G_rand = int(cnt0[m_rand][r_rand])
                delta_rand = L_rand - G_rand
                
                update_cnt(m_rand, r_old_rand, r_rand)
                
                current_uncovered += delta_rand
                best_residues[m_rand] = r_rand
            
            no_improve_steps = 0
    
    # 3. Final polish pass: exhaustive local search until no improvement
    # Recompute counts
    counts = np.zeros(N, dtype=np.int8)
    for m in divisors:
        counts[indices_cache[m][best_residues[m]]] += 1
    
    cnt1 = {}
    cnt0 = {}
    for m in divisors:
        cnt1[m] = np.zeros(m, dtype=np.int32)
        cnt0[m] = np.zeros(m, dtype=np.int32)
        for r in range(m):
            idx = indices_cache[m][r]
            cnt1[m][r] = np.sum(counts[idx] == 1)
            cnt0[m][r] = np.sum(counts[idx] == 0)
    
    current_uncovered = compute_uncovered(best_residues)
    
    improved = True
    polish_deadline = start_time + 118.0
    
    while improved and time.time() < polish_deadline:
        improved = False
        for m in divisors:
            if time.time() >= polish_deadline:
                break
            r_old = best_residues[m]
            L = int(cnt1[m][r_old])
            max_G = np.max(cnt0[m])
            if max_G > L:
                candidates = np.where(cnt0[m] == max_G)[0]
                candidates = candidates[candidates != r_old]
                if len(candidates) > 0:
                    # Pick the first one (deterministic for polish)
                    best_r = int(candidates[0])
                    best_delta = L - int(max_G)
                    
                    update_cnt(m, r_old, best_r)
                    
                    current_uncovered += best_delta
                    best_residues[m] = best_r
                    improved = True
    
    # Final check
    final_unc = compute_uncovered(best_residues)
    if final_unc < best_uncovered:
        best_uncovered = final_unc
    
    result = [[m, best_residues[m]] for m in divisors]
    return result
