"""Improved B&B MWIS with accurate bounds and enhanced local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction

def construct():
    deadline = time.time() + 54  # safety margin before 60s kill
    t0 = time.time()
    
    n_max = 256
    small = list(range(2, 129))  # 2..128
    large = list(range(129, 257))  # 129..256
    n_small = len(small)
    n_large = len(large)
    
    small_idx = {v: i for i, v in enumerate(small)}
    large_idx = {v: i for i, v in enumerate(large)}
    
    # Small weights
    sw = [Fraction(1, v) for v in small]
    # Large weights
    lw = [Fraction(1, v) for v in large]
    
    # Small-small conflict: lcm(small[i], small[j]) <= 256
    ss_conflict = [0] * n_small
    for i in range(n_small):
        for j in range(i + 1, n_small):
            if lcm(small[i], small[j]) <= 256:
                ss_conflict[i] |= (1 << j)
                ss_conflict[j] |= (1 << i)
    
    # Small-large conflict: small[i] divides large[j]
    sl_conflict = [0] * n_small
    for i, s in enumerate(small):
        for j, L in enumerate(large):
            if L % s == 0:
                sl_conflict[i] |= (1 << j)
    
    # Float weights for fast bound computation
    sw_float = [float(w) for w in sw]
    lw_float = [float(w) for w in lw]
    
    # Suffix small weights (float) - in original order
    suffix_sw_float = [0.0] * (n_small + 1)
    for i in range(n_small - 1, -1, -1):
        suffix_sw_float[i] = suffix_sw_float[i + 1] + sw_float[i]
    
    total_lw_float = sum(lw_float)
    
    # Precompute: for each small vertex, the float sum of large weights it excludes
    sl_excl_float = [0.0] * n_small
    for i in range(n_small):
        for j in range(n_large):
            if (sl_conflict[i] >> j) & 1:
                sl_excl_float[i] += lw_float[j]
    
    def compute_exact_score(mask, excluded):
        s = Fraction(0)
        for i in range(n_small):
            if (mask >> i) & 1:
                s += sw[i]
        for j in range(n_large):
            if not (excluded >> j) & 1:
                s += lw[j]
        return s
    
    # Multiple greedy seeds
    best_score = Fraction(0)
    best_small_mask = 0
    best_large_excluded = 0
    
    # Greedy 1: ascending order
    def greedy_small_order(order):
        selected = []
        for s in order:
            if all(lcm(s, y) > 256 for y in selected):
                selected.append(s)
        return selected
    
    # Try ascending
    greedy_sel = greedy_small_order(small)
    greedy_mask = 0
    greedy_excluded = 0
    for s in greedy_sel:
        i = small_idx[s]
        greedy_mask |= (1 << i)
        greedy_excluded |= sl_conflict[i]
    sc = compute_exact_score(greedy_mask, greedy_excluded)
    if sc > best_score:
        best_score = sc
        best_small_mask = greedy_mask
        best_large_excluded = greedy_excluded
    
    # Try descending
    greedy_sel2 = greedy_small_order(list(reversed(small)))
    greedy_mask2 = 0
    greedy_excluded2 = 0
    for s in greedy_sel2:
        i = small_idx[s]
        greedy_mask2 |= (1 << i)
        greedy_excluded2 |= sl_conflict[i]
    sc2 = compute_exact_score(greedy_mask2, greedy_excluded2)
    if sc2 > best_score:
        best_score = sc2
        best_small_mask = greedy_mask2
        best_large_excluded = greedy_excluded2
    
    # All large only
    all_large_score = Fraction(0)
    for j in range(n_large):
        all_large_score += lw[j]
    if all_large_score > best_score:
        best_score = all_large_score
        best_small_mask = 0
        best_large_excluded = 0
    
    # Random greedy seeds
    import random
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        remaining = list(range(n_small))
        random.shuffle(remaining)
        sel_mask = 0
        sel_excl = 0
        for i in remaining:
            if (sel_mask & ss_conflict[i]) == 0:
                sel_mask |= (1 << i)
                sel_excl |= sl_conflict[i]
        sc = compute_exact_score(sel_mask, sel_excl)
        if sc > best_score:
            best_score = sc
            best_small_mask = sel_mask
            best_large_excluded = sel_excl
    
    # Branch and bound with ACCURATE large availability tracking
    import sys
    sys.setrecursionlimit(10000)
    
    def solve_bnb():
        nonlocal best_score, best_small_mask, best_large_excluded, deadline
        
        # Order vertices by descending weight (ascending value) for tighter bounds
        order = sorted(range(n_small), key=lambda i: small[i])  # 2,3,4,...,128
        # Rearrange conflicts for this order
        # pos_order[k] = original index of k-th vertex in B&B order
        pos_order = order
        # For each B&B position k, the original index is pos_order[k]
        # ss_conflict_ordered[k] = bitmask of conflicts with other positions
        ss_conf_ordered = [0] * n_small
        for k in range(n_small):
            orig_i = pos_order[k]
            for m in range(n_small):
                if m != k:
                    orig_j = pos_order[m]
                    if (ss_conflict[orig_i] >> orig_j) & 1:
                        ss_conf_ordered[k] |= (1 << m)
        
        # Suffix sum in B&B order
        suffix_ordered = [0.0] * (n_small + 1)
        for k in range(n_small - 1, -1, -1):
            orig_i = pos_order[k]
            suffix_ordered[k] = suffix_ordered[k + 1] + sw_float[orig_i]
        
        # sl_conflict in B&B order: for position k, which large indices are excluded
        sl_conf_ordered = [sl_conflict[pos_order[k]] for k in range(n_small)]
        sl_excl_ordered = [sl_excl_float[pos_order[k]] for k in range(n_small)]
        sw_ordered = [sw_float[pos_order[k]] for k in range(n_small)]
        sw_frac_ordered = [sw[pos_order[k]] for k in range(n_small)]
        
        def dfs(pos, selected_mask, excluded_large, current_score, current_float, avail_large_float):
            nonlocal best_score, best_small_mask, best_large_excluded, deadline
            
            if time.time() > deadline:
                return
            
            if pos == n_small:
                # Compute exact score
                large_score = Fraction(0)
                for j in range(n_large):
                    if not (excluded_large >> j) & 1:
                        large_score += lw[j]
                total = current_score + large_score
                if total > best_score:
                    best_score = total
                    # Convert back to original mask
                    orig_mask = 0
                    for k in range(n_small):
                        if (selected_mask >> k) & 1:
                            orig_mask |= (1 << pos_order[k])
                    best_small_mask = orig_mask
                    best_large_excluded = excluded_large
                return
            
            # Upper bound: current + remaining smalls + available large
            ub = current_float + suffix_ordered[pos] + avail_large_float
            if ub <= float(best_score) + 1e-12:
                return
            
            orig_i = pos_order[pos]
            
            # Option 1: Include vertex pos (if compatible)
            if (selected_mask & ss_conf_ordered[pos]) == 0:
                new_mask = selected_mask | (1 << pos)
                new_excluded = excluded_large | sl_conf_ordered[pos]
                # Accurate marginal loss: only subtract large weights that are NEWLY excluded
                new_excl_bits = sl_conf_ordered[pos] & ~excluded_large
                marginal_loss = 0.0
                # Count bits and sum weights
                bits = new_excl_bits
                while bits:
                    lb = bits & (-bits)
                    j = lb.bit_length() - 1
                    marginal_loss += lw_float[j]
                    bits ^= lb
                new_avail = avail_large_float - marginal_loss
                new_float = current_float + sw_ordered[pos]
                new_score = current_score + sw_frac_ordered[pos]
                
                dfs(pos + 1, new_mask, new_excluded, new_score, new_float, new_avail)
            
            # Option 2: Exclude vertex pos
            dfs(pos + 1, selected_mask, excluded_large, current_score, current_float, avail_large_float)
        
        dfs(0, 0, 0, Fraction(0), 0.0, total_lw_float)
    
    solve_bnb()
    
    # Enhanced local search
    if time.time() < deadline - 5:
        improved = True
        max_rounds = 50
        rnd = 0
        while improved and time.time() < deadline - 3 and rnd < max_rounds:
            improved = False
            rnd += 1
            
            # Try adding each non-selected small
            for i in range(n_small):
                if (best_small_mask >> i) & 1:
                    continue
                if (best_small_mask & ss_conflict[i]) != 0:
                    continue
                new_mask = best_small_mask | (1 << i)
                new_excl = best_large_excluded | sl_conflict[i]
                new_score = compute_exact_score(new_mask, new_excl)
                if new_score > best_score:
                    best_score = new_score
                    best_small_mask = new_mask
                    best_large_excluded = new_excl
                    improved = True
            
            # Try removing each selected small
            for i in range(n_small):
                if not (best_small_mask >> i) & 1:
                    continue
                new_mask = best_small_mask & ~(1 << i)
                new_excl = 0
                for k in range(n_small):
                    if (new_mask >> k) & 1:
                        new_excl |= sl_conflict[k]
                new_score = compute_exact_score(new_mask, new_excl)
                if new_score > best_score:
                    best_score = new_score
                    best_small_mask = new_mask
                    best_large_excluded = new_excl
                    improved = True
            
            # Try 1-in-1-out exchanges
            for i_out in range(n_small):
                if not (best_small_mask >> i_out) & 1:
                    continue
                for i_in in range(n_small):
                    if (best_small_mask >> i_in) & 1:
                        continue
                    temp_mask = best_small_mask & ~(1 << i_out)
                    if (temp_mask & ss_conflict[i_in]) != 0:
                        continue
                    new_mask = temp_mask | (1 << i_in)
                    new_excl = 0
                    for k in range(n_small):
                        if (new_mask >> k) & 1:
                            new_excl |= sl_conflict[k]
                    new_score = compute_exact_score(new_mask, new_excl)
                    if new_score > best_score:
                        best_score = new_score
                        best_small_mask = new_mask
                        best_large_excluded = new_excl
                        improved = True
                        break
                if improved:
                    break
    
    # Build final set
    result = []
    for i in range(n_small):
        if (best_small_mask >> i) & 1:
            result.append(small[i])
    for j in range(n_large):
        if not (best_large_excluded >> j) & 1:
            result.append(large[j])
    
    if not result:
        return [2]
    
    # Verify constraint
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if lcm(result[i], result[j]) <= 256:
                return [2]
    
    return result
# EVOLVE-BLOCK-END
