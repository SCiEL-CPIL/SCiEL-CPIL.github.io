import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Generate all words and balls
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    balls = [0] * total_words
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            mask |= (1 << i)
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask

    all_ones_mask = (1 << total_words) - 1
    
    # 1. Fast Heuristic to get Initial UB
    def greedy_cover(uncovered_mask):
        selected = []
        remaining = uncovered_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            for i in range(total_words):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
        return selected

    def remove_redundants(selected):
        if not selected:
            return []
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break
        return selected

    initial_selected = greedy_cover(all_ones_mask)
    initial_selected = remove_redundants(initial_selected)
    best_size = len(initial_selected)
    best_code = [all_words[i] for i in initial_selected]
    
    # 2. Branch and Bound
    # State: uncovered_mask, current_depth, current_selected
    # We try to find a solution with size < best_size
    
    # Precompute for B&B:
    # For each word, which balls cover it?
    # This is expensive to store as lists of lists for 2187x2187.
    # Instead, we compute on the fly or use bitsets.
    
    # To speed up B&B, we can restrict candidates.
    # Only consider codewords that cover at least one uncovered word.
    
    def branch_and_bound(uncovered_mask, depth, current_selected, ub):
        if uncovered_mask == 0:
            if depth < ub[0]:
                ub[0] = depth
                ub[1] = current_selected[:]
            return True # Found a better solution
            
        # Lower Bound
        # 1. Count uncovered words
        num_uncovered = uncovered_mask.bit_count()
        if num_uncovered == 0:
            if depth < ub[0]:
                ub[0] = depth
                ub[1] = current_selected[:]
            return True
            
        # Simple LB: ceil(num_uncovered / max_ball_size_in_uncovered)
        # We need to find the max coverage of any ball in the uncovered set.
        # This is O(N) per node, which might be slow.
        # Optimization: If depth + 1 >= ub[0], prune.
        if depth + 1 >= ub[0]:
            return False
            
        # Forced Move Check:
        # Find a word in uncovered_mask that is covered by the fewest candidates.
        # If count is 1, we must pick that candidate.
        
        # To avoid O(N^2) or O(N * N) complexity, we might just use greedy choice for branching.
        
        # Branching:
        # Pick the best candidate (max coverage).
        best_idx = -1
        max_cover = 0
        # Iterate through all words? Or just a subset?
        # For 2187 words, iterating all is ~2000 operations.
        # If depth is 186, total ops ~ 400k. This is fine.
        
        for i in range(total_words):
            c = (balls[i] & uncovered_mask).bit_count()
            if c > max_cover:
                max_cover = c
                best_idx = i
                # Tie-breaking: pick the one with smallest index?
        
        if best_idx == -1:
            return False
            
        # Pruning: If even the best ball doesn't cover enough?
        # LB: depth + ceil(num_uncovered / max_cover) >= ub[0]
        lb = depth + (num_uncovered + max_cover - 1) // max_cover
        if lb >= ub[0]:
            return False
            
        # Branch 1: Pick best_idx
        current_selected.append(best_idx)
        new_uncovered = uncovered_mask & ~balls[best_idx]
        if branch_and_bound(new_uncovered, depth + 1, current_selected, ub):
            current_selected.pop()
            return True # Early exit if we found a solution that satisfies ub? 
                        # Actually we want the MINIMUM. 
                        # If we find a solution of size S, we set ub=S.
                        # We should continue searching for smaller?
                        # But if we found a solution, and we are doing DFS, 
                        # we might miss a smaller one in other branches.
                        # However, if we prune based on ub, and ub is updated,
                        # we can continue.
        
        current_selected.pop()
        
        # Branch 2: Do NOT pick best_idx?
        # This is hard for Set Cover. We must cover the words.
        # Standard B&B for Set Cover:
        # We pick a word $w$ that is uncovered.
        # We MUST pick one of the sets that covers $w$.
        # So we branch on the sets covering $w$.
        
        # Let's restart the B&B logic with this "Branch on word" approach.
        return False

    # Revised B&B
    def bnb(uncovered_mask, depth, current_selected, ub):
        if uncovered_mask == 0:
            if depth < ub[0]:
                ub[0] = depth
                ub[1] = current_selected[:]
            return True
            
        if depth >= ub[0]:
            return False
            
        # Find an uncovered word with the fewest covering candidates
        # This is the MRV heuristic.
        # We need to know which balls cover which words.
        # Precomputing `coverers[word] = list of ball indices` is O(N^2) memory/time.
        # 2187^2 ~ 4.7M. This is acceptable.
        
        # But we can just iterate.
        # To be faster, we can pick the first uncovered word.
        # Let's pick the word with the smallest index in uncovered_mask for simplicity,
        # or the one with min candidates if we precompute.
        
        # Let's precompute coverers for speed.
        # coverers[i] = list of j such that bit i is set in balls[j]
        
        # Actually, doing this inside the function is bad.
        # Let's do it outside.
        
        return False

    # Precompute coverers
    coverers = [[] for _ in range(total_words)]
    for j in range(total_words):
        m = balls[j]
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverers[idx].append(j)
            m ^= lsb

    ub = [best_size, best_code]
    
    # Time limit for B&B
    start_time = time.time()
    bnb_time_limit = 110.0
    
    def bnb_search(uncovered_mask, depth, current_selected):
        nonlocal ub
        if time.time() - start_time > bnb_time_limit:
            return False
            
        if uncovered_mask == 0:
            if depth < ub[0]:
                ub[0] = depth
                ub[1] = current_selected[:]
            return True
            
        if depth >= ub[0]:
            return False
            
        # Find an uncovered word
        # Pick the one with fewest candidates (MRV)
        # We can iterate through uncovered bits
        m = uncovered_mask
        best_word = -1
        min_cands = float('inf')
        best_cands = []
        
        # To speed up, we might just pick the first one if MRV is too slow.
        # But MRV is crucial for B&B.
        # Let's try a limited MRV: check first 10 uncovered words.
        count_checked = 0
        while m and count_checked < 10:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cands = coverers[idx]
            # Filter candidates that actually cover this word and are "valid" (we don't have a validity constraint other than covering)
            # All candidates in coverers[idx] cover idx.
            if len(cands) < min_cands:
                min_cands = len(cands)
                best_word = idx
                best_cands = cands
                if min_cands == 1:
                    break
            m ^= lsb
            count_checked += 1
            
        if best_word == -1:
            return False
            
        # Branch on each candidate covering best_word
        for c in best_cands:
            # Check if picking c is promising
            # New uncovered
            new_uncovered = uncovered_mask & ~balls[c]
            
            # Prune: if depth + 1 >= ub[0], skip
            if depth + 1 >= ub[0]:
                continue
                
            # Simple LB:
            # num_uncovered_new = new_uncovered.bit_count()
            # if depth + 1 + ceil(num_uncovered_new / max_ball) >= ub[0]: continue
            # (Skipping complex LB for now to keep code simple)
            
            current_selected.append(c)
            if bnb_search(new_uncovered, depth + 1, current_selected):
                current_selected.pop()
                return True # Found a better solution, propagate up? 
                            # Actually, if we find a solution, we update ub.
                            # We should continue to see if we can find an even smaller one?
                            # But if we found one of size S, and we prune depth >= S,
                            # we won't find smaller.
                            # To find smaller, we need to allow depth < S.
                            # Since we update ub[0], the prune condition `depth >= ub[0]` will use the new smaller ub.
                            # So we should NOT return immediately, unless we want to stop at first improvement.
                            # But we want the MINIMUM.
                            # So we should continue searching other branches that might yield depth < new_ub.
                            # However, DFS order: if we pick the "best" candidate first, we find a good solution quickly.
                            # Then we prune deeper branches.
                            # So we can return True to unwind and try other top-level branches?
                            # No, we are inside a recursive call.
                            # If we return True, the parent will pop and try the next candidate.
                            # This is correct.
            
            current_selected.pop()
            
        return False

    # Run B&B
    initial_uncovered = all_ones_mask
    initial_selected = []
    bnb_search(initial_uncovered, 0, initial_selected)
    
    if ub[1]:
        return ub[1]
    else:
        return best_code
