import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1

    # Tetracode: 9 words covering all 81 words of length 4 over {0,1,2} with radius 1
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    
    # Convert tetracode to integer indices (base-3)
    tetracode_idx = []
    for tc in tetracode:
        val = 0
        for d in tc:
            val = val * 3 + d
        tetracode_idx.append(val)
    
    # Partition-based construction:
    # Split 8 positions into first 4 (head) and last 4 (tail)
    # For each of the 3^4 = 81 possible heads, we need to cover all 81 tails
    # with words whose head matches or is at distance 1 from the given head
    
    # Strategy: For each head h, select a set of (head', tail') pairs where head' 
    # is within distance 1 of h, such that all tails are covered.
    
    # Precompute: for each head, which heads are within distance 1
    head_neighbors = {}
    for h in range(81):
        # Decode head
        digits = []
        temp = h
        for _ in range(4):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        
        # Find all heads within distance 1
        hn = [h]
        for pos in range(4):
            weight = 3 ** (3 - pos)
            for new_d in range(3):
                if new_d == digits[pos]:
                    continue
                new_h = h + (new_d - digits[pos]) * weight
                hn.append(new_h)
        head_neighbors[h] = hn
    
    # Precompute: for each tail t, which tails are within distance 1
    tail_neighbors = {}
    for t in range(81):
        digits = []
        temp = t
        for _ in range(4):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        
        tn = [t]
        for pos in range(4):
            weight = 3 ** (3 - pos)
            for new_d in range(3):
                if new_d == digits[pos]:
                    continue
                new_t = t + (new_d - digits[pos]) * weight
                tn.append(new_t)
        tail_neighbors[t] = tn
    
    # Convert head and tail indices to full 8-digit indices
    def head_to_full(head, tail):
        return head * 81 + tail
    
    # For each head h, we need to choose codewords (h', t') such that:
    # 1. h' is within distance 1 of h
    # 2. The set of tails {t' : (h', t') selected for h} covers all 81 tails
    #    (i.e., for every tail t, there exists selected (h', t') with t' within distance 1 of t)
    #    BUT ALSO: codewords selected for different heads can overlap in coverage
    
    # Greedy approach: For each head, pick codewords that maximize new coverage
    # Start with empty selection, add codewords one by one
    
    # Actually, let's use a simpler initial construction:
    # For each head h, use the tetracode structure on the 5 heads within distance 1
    # But that's complex. Let's start with a simple heuristic:
    # For each head h, select all 9 tetracode words with head = h (i.e., 9 words per head)
    # Total: 81 * 9 = 729 words. Then prune aggressively.
    
    # Better: Use the fact that codewords with head h' cover heads within distance 1 of h'
    # So a codeword (h', t') covers all (h, t) where dist(h, h') <= 1 and dist(t, t') <= 1
    
    # Initial construction: For each head h, pick codewords that help cover all heads
    # Let's use a randomized greedy per head
    
    best_size = total
    best_set = None
    
    # Precompute: for each possible codeword index, which (head, tail) pairs it covers
    # codeword c = head_c * 81 + tail_c
    # It covers (h, t) if dist(head_c, h) <= 1 and dist(tail_c, t) <= 1
    
    # For efficiency, precompute for each head_c, the set of heads it covers
    head_covered_by = {}
    for hc in range(81):
        covered = set()
        for h in head_neighbors[hc]:
            covered.add(h)
        head_covered_by[hc] = covered
    
    # For each tail_c, the set of tails it covers
    tail_covered_by = {}
    for tc in range(81):
        covered = set()
        for t in tail_neighbors[tc]:
            covered.add(t)
        tail_covered_by[tc] = covered
    
    while time.time() < deadline - 5.0:
        # Randomized greedy construction using partition structure
        # We'll select codewords one by one, prioritizing those that cover many uncovered (head, tail) pairs
        
        # Represent uncovered as a 2D boolean array [81][81]
        uncovered = np.ones((81, 81), dtype=bool)
        uncovered_count = 81 * 81
        
        selected = []
        in_selected = np.zeros(total, dtype=bool)
        
        while uncovered_count > 0:
            if time.time() > deadline - 10.0:
                break
            
            # For each candidate codeword c = (hc, tc), compute how many uncovered (h, t) it covers
            # A codeword (hc, tc) covers (h, t) if h in head_neighbors[hc] and t in tail_neighbors[tc]
            
            # We want to find the codeword that covers the most uncovered pairs
            # To speed this up, we can precompute for each (hc, tc) the count
            
            # Let's compute counts for all 6561 codewords
            # counts[c] = sum(uncovered[h, t] for h in head_neighbors[hc] for t in tail_neighbors[tc])
            
            # This is expensive to compute from scratch each time. 
            # Alternative: maintain a 2D array and use separability.
            
            # For a given (hc, tc):
            # count = sum_{h in HN(hc)} sum_{t in TN(tc)} uncovered[h, t]
            # = sum_{h in HN(hc)} row_sums[h] where row_sums depends on tc
            
            # Let's precompute for each tc: for each h, the number of uncovered t in TN(tc)
            # Then for each hc: sum over h in HN(hc) of that value
            
            # Step 1: For each tc in 0..80, compute col_sums_tc[h] = sum_{t in TN(tc)} uncovered[h, t]
            # This is 81 * 81 * 9 operations = ~60000, which is fine
            
            # Actually, let's just compute the full 6561 counts each iteration. 
            # With 81*81=6561 uncovered pairs max, and 6561 codewords, it's 6561*6561 ~ 43M ops worst case.
            # That's too slow. Let's be smarter.
            
            # Smarter: For each codeword c=(hc,tc), the count is:
            # sum_{h in HN(hc)} sum_{t in TN(tc)} uncovered[h,t]
            
            # Precompute: for each h, row_sum[h] = sum_t uncovered[h,t]  (81 values)
            # Precompute: for each t, col_sum[t] = sum_h uncovered[h,t]  (81 values)
            
            # But the count for (hc,tc) is NOT simply related to row_sum and col_sum
            # because it's a 2D sum over a specific rectangle.
            
            # Let's use a different approach: 
            # For each hc, precompute H_mask[hc][h] = 1 if h in HN(hc)
            # For each tc, precompute T_mask[tc][t] = 1 if t in TN(tc)
            
            # Then count(hc, tc) = sum_{h,t} uncovered[h,t] * H_mask[hc][h] * T_mask[tc][t]
            
            # This is a bilinear form. We can compute it as:
            # Let M = uncovered * H_mask[hc]  (for each hc)
            # Then count(hc, tc) = sum_t M[tc_row] * T_mask[tc]... no.
            
            # Actually: count(hc, tc) = sum_h (sum_t uncovered[h,t] * T_mask[tc][t]) * H_mask[hc][h]
            # = sum_h (uncovered[h,:] @ T_mask[tc]) * H_mask[hc][h]
            
            # So: 
            # 1. For each tc, compute V_tc[h] = uncovered[h,:] @ T_mask[tc]  (81*81*81 = 531K ops)
            # 2. For each hc, count(hc,tc) = V_tc @ H_mask[hc]  (81*81*81 = 531K ops)
            # Total: ~1M ops per iteration. With ~500 iterations, that's 500M ops. Might be slow.
            
            # Let's just do a simpler greedy: pick a random uncovered (h,t), 
            # then pick a codeword that covers it, preferring those that cover many uncovered.
            
            # Pick a random uncovered position
            uncov_idx = np.where(uncovered)
            if len(uncov_idx[0]) == 0:
                break
            r = np.random.randint(len(uncov_idx[0]))
            h_rand = uncov_idx[0][r]
            t_rand = uncov_idx[1][r]
            
            # Find all codewords that cover (h_rand, t_rand)
            # These are codewords (hc, tc) where h_rand in HN(hc) and t_rand in TN(tc)
            # i.e., hc in HN(h_rand) and tc in TN(t_rand)
            
            # HN(h_rand) is the set of heads within distance 1 of h_rand
            # TN(t_rand) is the set of tails within distance 1 of t_rand
            
            # For each (hc, tc) in HN(h_rand) x TN(t_rand), compute how many uncovered it covers
            # and pick the best one
            
            hc_candidates = head_neighbors[h_rand]
            tc_candidates = tail_neighbors[t_rand]
            
            best_c = -1
            best_count = -1
            best_hc = -1
            best_tc = -1
            
            for hc in hc_candidates:
                for tc in tc_candidates:
                    c = hc * 81 + tc
                    if in_selected[c]:
                        continue
                    
                    # Count uncovered covered by this codeword
                    cnt = 0
                    for h in head_neighbors[hc]:
                        for t in tail_neighbors[tc]:
                            if uncovered[h, t]:
                                cnt += 1
                    
                    if cnt > best_count:
                        best_count = cnt
                        best_c = c
                        best_hc = hc
                        best_tc = tc
            
            if best_c == -1:
                # No non-selected codeword covers this position? Shouldn't happen.
                break
            
            # Select best_c
            selected.append(best_c)
            in_selected[best_c] = True
            
            # Update uncovered
            for h in head_neighbors[best_hc]:
                for t in tail_neighbors[best_tc]:
                    if uncovered[h, t]:
                        uncovered[h, t] = False
                        uncovered_count -= 1
        
        if uncovered_count > 0:
            continue
        
        # Local search: try to remove codewords
        # A codeword c=(hc,tc) is removable if for every (h,t) it covers, 
        # there's another selected codeword covering (h,t)
        
        # Build cover count: for each (h,t), how many selected codewords cover it
        cover_count = np.zeros((81, 81), dtype=np.int32)
        for c in selected:
            hc = c // 81
            tc = c % 81
            for h in head_neighbors[hc]:
                for t in tail_neighbors[tc]:
                    cover_count[h, t] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            
            # Try to remove a codeword
            # A codeword c is removable if min(cover_count over its covered (h,t)) > 1
            # i.e., every (h,t) it covers is covered by at least one other codeword
            
            removable = []
            for i, c in enumerate(selected):
                hc = c // 81
                tc = c % 81
                min_cc = 999
                for h in head_neighbors[hc]:
                    for t in tail_neighbors[tc]:
                        if cover_count[h, t] < min_cc:
                            min_cc = cover_count[h, t]
                if min_cc > 1:
                    removable.append(i)
            
            if removable:
                # Remove one randomly
                idx = removable[np.random.randint(len(removable))]
                c = selected[idx]
                hc = c // 81
                tc = c % 81
                for h in head_neighbors[hc]:
                    for t in tail_neighbors[tc]:
                        cover_count[h, t] -= 1
                selected.pop(idx)
                in_selected[c] = False
                changed = True
                continue
            
            # Try swaps: remove one, see if we can add another that allows removal of two
            # For now, just try random swaps
            if len(selected) > 486:
                idx = np.random.randint(len(selected))
                c_remove = selected[idx]
                hc_r = c_remove // 81
                tc_r = c_remove % 81
                
                # Temporarily remove
                for h in head_neighbors[hc_r]:
                    for t in tail_neighbors[tc_r]:
                        cover_count[h, t] -= 1
                selected.pop(idx)
                in_selected[c_remove] = False
                
                # Find uncovered (h,t)
                U = np.where(cover_count == 0)
                if len(U[0]) > 0:
                    # Need a codeword that covers all uncovered positions
                    # Check if any non-selected codeword covers all U
                    # This is expensive, so skip for now
                    pass
                
                # Revert
                selected.insert(idx, c_remove)
                in_selected[c_remove] = True
                for h in head_neighbors[hc_r]:
                    for t in tail_neighbors[tc_r]:
                        cover_count[h, t] += 1
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486:
                break
        
        # Random restart with perturbation
        if best_set is not None and random.random() < 0.3:
            # Start from best_set and perturb
            pass
    
    if best_set is None:
        # Fallback: return all words
        return [list(np.base_repr(w, 3).zfill(n).ljust(n, '0')) for w in range(total)]
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
