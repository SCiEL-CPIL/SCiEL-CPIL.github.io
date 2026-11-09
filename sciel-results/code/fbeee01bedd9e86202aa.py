"""Covering code K_q(n,1) via bitset-based greedy set-cover + local search."""
import time
import random
import itertools
import numpy as np

_DEADLINE = 115.0  # seconds; keep safety margin under the 120 s kill


def _time_left(deadline):
    return deadline - time.time()


def construct(q, n, radius):
    deadline = time.time() + _DEADLINE
    rng = random.Random(12345)

    if radius != 1:
        # Fallback: full space
        return [list(c) for c in itertools.product(range(q), repeat=n)]

    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute balls as Python integers (bitsets)
    # bit i is 1 if word i is in the ball
    balls = [0] * total
    for w in all_words:
        idx = word_to_idx[w]
        mask = 1 << idx
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nidx = word_to_idx[tuple(nw)]
                mask |= (1 << nidx)
        balls[idx] = mask

    # Greedy Set Cover
    uncovered = (1 << total) - 1
    chosen = []
    chosen_set = set()
    
    while uncovered and _time_left(deadline) > 5.0:
        best_idx = -1
        best_count = -1
        # Optimization: If total is small, scanning all is fine.
        # 2187 candidates * popcount (fast) is very quick.
        for idx in range(total):
            if idx in chosen_set:
                continue
            cov = (balls[idx] & uncovered).bit_count()
            if cov > best_count:
                best_count = cov
                best_idx = idx
                if cov == uncovered.bit_count():
                    break
        if best_idx < 0:
            break
        chosen.append(best_idx)
        chosen_set.add(best_idx)
        uncovered &= ~balls[best_idx]

    # Local Search
    # Maintain cover_mask for quick checks? 
    # Actually, for redundancy, we need to know if ball[i] is covered by others.
    # others_mask = OR of balls[j] for j in chosen, j != i.
    
    def _remove_redundant(chosen_list):
        changed = True
        while changed and _time_left(deadline) > 3.0:
            changed = False
            n_ch = len(chosen_list)
            for i in range(n_ch):
                # Calculate OR of all other balls
                others_mask = 0
                for j in range(n_ch):
                    if j != i:
                        others_mask |= balls[chosen_list[j]]
                
                # If ball[i] is subset of others_mask, it's redundant
                if (balls[chosen_list[i]] & ~others_mask) == 0:
                    chosen_list.pop(i)
                    changed = True
                    break
        return chosen_list

    chosen_list = list(chosen)
    chosen_list = _remove_redundant(chosen_list)
    
    # Iterative Improvement
    # Try to replace a codeword with another to enable removal of a third, or just to reduce "unique" burden.
    improved = True
    while improved and _time_left(deadline) > 3.0:
        improved = False
        chosen_list = _remove_redundant(chosen_list)
        n_ch = len(chosen_list)
        if n_ch == 0:
            break
        
        # Find the codeword with the smallest unique contribution
        min_unique_count = float('inf')
        min_idx = -1
        min_unique_mask = 0
        min_others_mask = 0
        
        for i in range(n_ch):
            others_mask = 0
            for j in range(n_ch):
                if j != i:
                    others_mask |= balls[chosen_list[j]]
            
            unique_mask = balls[chosen_list[i]] & ~others_mask
            count = unique_mask.bit_count()
            
            if count < min_unique_count:
                min_unique_count = count
                min_idx = i
                min_unique_mask = unique_mask
                min_others_mask = others_mask
                
                if count == 0:
                    break # Should have been caught by _remove_redundant, but safety
            
        if min_unique_count == 0:
            # Redundant codeword found, remove it
            chosen_list.pop(min_idx)
            improved = True
            continue
            
        # Try to find a replacement j not in chosen_list
        # such that ball[j] covers min_unique_mask
        # And ideally, the new set allows removing another codeword.
        # For now, just replace if it doesn't increase size and allows future reduction.
        # A good heuristic: Replace i with j if j covers unique_i and j is "better" (e.g., covers more of the "border").
        
        best_j = -1
        best_score = -1
        
        # Only check candidates that cover the unique mask
        # To speed up, we can precompute which balls cover a given mask? No, mask is dynamic.
        # Just iterate all non-chosen.
        for j in range(total):
            if j in chosen_set: # Note: chosen_set is stale if we modified chosen_list. 
                # We need to update chosen_set or use a set from chosen_list
                pass
            
        # Rebuild chosen_set for speed
        current_chosen_set = set(chosen_list)
        
        for j in range(total):
            if j in current_chosen_set:
                continue
            if (balls[j] & min_unique_mask) == min_unique_mask:
                # j covers all unique words of i
                # Score: How many words does j cover that are NOT covered by others_mask?
                # If j covers many new words, it's good.
                new_cov = (balls[j] & ~min_others_mask).bit_count()
                if new_cov > best_score:
                    best_score = new_cov
                    best_j = j
                    
        if best_j != -1:
            # Replace i with j
            chosen_list[min_idx] = best_j
            improved = True
            # Loop will restart, calling _remove_redundant again
            continue

    # Final cleanup
    chosen_list = _remove_redundant(chosen_list)
    
    # Verify validity
    cover = 0
    for idx in chosen_list:
        cover |= balls[idx]
    if cover != (1 << total) - 1:
        # Fallback to greedy result
        chosen_list = list(chosen)
        chosen_list = _remove_redundant(chosen_list)

    return [list(all_words[idx]) for idx in chosen_list]
