import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases (not expected in this specific task instance but good for robustness)
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
    
    # Precompute balls
    # weights for positions
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    # balls[:, 0] is the word itself
    # balls[:, 1 + 2*pos : 2 + 2*pos+1] are the neighbors?
    # Actually, let's just fill it.
    # For each word w, ball is [w] + [w + (d - w_d) * weight for pos in range(n) for d in range(3) if d != w_d]
    
    # Vectorized precomputation
    words = np.arange(total, dtype=np.int32)
    # Get digits
    # This is a bit tricky to vectorize fully without loops, but 6561 is small.
    # Let's use a loop for precomputation, it's fast enough once.
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        temp = w
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    best_size = total
    best_set = None
    
    while time.time() < deadline - 2.0:
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            # Count how many uncovered words each candidate covers
            # uncovered_mask[balls] shape (total, 17)
            # sum along axis 1
            counts = uncovered_mask[balls].sum(axis=1)
            
            # Find max count
            max_count = np.max(counts)
            
            # If max_count is 0, we are stuck (should not happen if space is connected)
            if max_count == 0:
                break
            
            # Find all candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Pick one randomly
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            # A word is uncovered if cover_count is 0
            # We can just recompute uncovered_mask = (cover_count == 0)
            # Or update it: for w in balls[best_idx], if cover_count[w] was 0, it is now 1, so uncovered.
            # But cover_count[w] could be >1 if it was already covered.
            # So: new_uncovered = (cover_count == 0) is safest.
            uncovered_mask = (cover_count == 0)
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        if not np.all(uncovered_mask == False):
            # If we broke early due to time, this set might be invalid.
            # We should only consider it if it's valid.
            # But checking validity takes time.
            # If we broke out of while loop normally, it is valid.
            # Let's check if we completed the loop.
            if np.any(uncovered_mask):
                continue
        
        # 2. Pruning
        # Convert selected to a set for fast lookup? Or just a list.
        # We need to check if removing a codeword keeps the set valid.
        # Condition: for all w in ball(c), cover_count[w] > 1.
        
        # We can iterate through selected and try to remove.
        # To be efficient, we can do multiple passes.
        
        # Make a copy of selected to iterate over, as we modify it.
        # Actually, we can just loop through indices.
        
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            # Iterate backwards to allow removal without index shifting issues?
            # Or just build a new list.
            new_selected = []
            for c in selected:
                # Check if c is removable
                # min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    # Remove c
                    cover_count[balls[c]] -= 1
                    changed = True
                    # Do not add to new_selected
                else:
                    new_selected.append(c)
            selected = new_selected
            
            # If we removed some, we might have made others removable?
            # Yes, so we loop again.
            
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            # If we found 486, we can stop early?
            # The lower bound is 403, best known 486.
            # It's unlikely to go below 486 quickly, but possible.
            # Let's keep searching.
            
        if best_size <= 486:
            # We found a very good solution.
            # We can try a bit more to see if we can go lower, but 486 is the target.
            # Let's run a few more iterations just in case.
            pass

    # Convert best_set to list of digit lists
    if best_set is None:
        # Fallback: return all words (invalid for score but valid format)
        return [list(np.base_repr(w, 3).zfill(n).ljust(n, '0')) for w in range(total)]
        # Wait, base_repr might not handle leading zeros well or might be slow.
        # Use the to_digits helper.
        
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
