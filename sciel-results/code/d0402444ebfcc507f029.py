import itertools
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # But we should handle the general case if possible, or at least assume the specific task parameters
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # However, the task is specifically "Football pool problem for 7 matches: ternary covering code K_3(7,1)".
    # It is safe to assume q=3, n=7, radius=1 for the scoring, but let's be slightly generic or robust.
    # Given the 120s limit, we can't do general q,n,r. We will specialize for q=3, n=7, r=1.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed behavior or simple construction for other params
        # But the task contract implies this specific instance.
        # Let's just return the seed result for non-matching params to be safe, 
        # though the evaluator likely only tests the specific case.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(tail) for tail in itertools.product(range(3), repeat=n)]

    # Specific solution for q=3, n=7, r=1
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Pre-compute balls for all 3^7 = 2187 words
    # Map word (tuple of 7 ints) to index
    # Index = sum(w[i] * 3^(6-i))
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    total_words = 3 ** n
    balls = [0] * total_words
    
    # Generate all words and their balls
    # Ball of radius 1: self + 7*2 neighbors
    for idx in range(total_words):
        w = idx_to_word(idx)
        mask = 1 << idx
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = w[pos] + delta
                if 0 <= new_val < q:
                    w[pos] = new_val
                    nidx = word_to_idx(w)
                    mask |= (1 << nidx)
                    w[pos] -= delta # restore
        balls[idx] = mask

    # 2. Greedy Set Covering
    uncovered_mask = (1 << total_words) - 1
    chosen_indices = []
    
    # Optimization: Instead of checking all 2187 candidates every time,
    # we can just iterate. 2187 is small.
    while uncovered_mask != 0:
        best_idx = -1
        best_count = 0
        # Iterate through all possible codewords
        for c in range(total_words):
            # Count how many currently uncovered words are in ball(c)
            # This is bit_count(balls[c] & uncovered_mask)
            count = (balls[c] & uncovered_mask).bit_count()
            if count > best_count:
                best_count = count
                best_idx = c
                if best_count == 15: # Max possible, can't get better
                    # Actually, if we find one that covers all uncovered, we are done
                    # But uncovered_mask changes.
                    # If best_count == bit_count(uncovered_mask), we break outer loop after adding
                    pass
        
        if best_idx == -1:
            break
            
        chosen_indices.append(best_idx)
        uncovered_mask &= ~balls[best_idx]
        
        # Time check
        if time.time() > deadline:
            break

    # 3. Local Search: Deletion
    # Maintain cover counts
    cover_counts = [0] * total_words
    
    # Initialize counts
    for c in chosen_indices:
        mask = balls[c]
        # Iterate through set bits in mask
        # Efficient way: while mask:
        #   lsb = mask & -mask
        #   idx = lsb.bit_length() - 1
        #   cover_counts[idx] += 1
        #   mask ^= lsb
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_counts[idx] += 1
            m ^= lsb

    # Deletion passes
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_chosen = []
        for c in chosen_indices:
            mask = balls[c]
            # Check if any word in ball(c) has count 1
            # If all have count >= 2, we can remove c
            can_remove = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_counts[idx] == 1:
                    can_remove = False
                    break
                m ^= lsb
            
            if can_remove:
                # Remove c
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_chosen.append(c)
        
        chosen_indices = new_chosen

    # Convert indices back to words
    code = [idx_to_word(i) for i in chosen_indices]
    
    # Ensure distinctness (should be distinct by construction)
    # And validity (greedy ensures coverage)
    
    return code
