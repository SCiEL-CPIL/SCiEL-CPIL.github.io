"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Only handle the specific case efficiently, but keep general interface
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases (though task is specific)
        # Simple greedy for general case
        all_words = list(range(q ** n))
        # Precompute balls for general case? Might be too slow for large n
        # For now, return a trivial valid code (all words) if not the specific case
        # But the task is specific to 3,6,1. We assume this is the only case.
        # If we must be general, we can do a basic greedy.
        def _hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        def _word_to_idx(w):
            idx = 0
            for v in w:
                idx = idx * q + v
            return idx
            
        def _idx_to_word(idx):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w

        covered = set()
        codewords = []
        # Greedy: pick word covering most uncovered
        # This is O(N^2 * n) which is fine for small N, but 3^6=729 is small.
        uncovered = set(range(q ** n))
        while uncovered:
            best_idx = -1
            best_count = -1
            for idx in uncovered:
                w = _idx_to_word(idx)
                # Count how many uncovered words are in ball of w
                count = 0
                # Ball of radius 1
                for d in range(n):
                    for val in range(q):
                        if val == w[d]: continue
                        nw = w.copy()
                        nw[d] = val
                        ni = _word_to_idx(nw)
                        if ni in uncovered:
                            count += 1
                if w in uncovered:
                    count += 1
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1: break
            codewords.append(_idx_to_word(best_idx))
            # Update uncovered
            w = _idx_to_word(best_idx)
            to_remove = set()
            to_remove.add(best_idx)
            for d in range(n):
                for val in range(q):
                    if val == w[d]: continue
                    nw = w.copy()
                    nw[d] = val
                    ni = _word_to_idx(nw)
                    to_remove.add(ni)
            uncovered -= to_remove
            if time.time() > deadline:
                break
        return codewords

    # Specific case: q=3, n=6, radius=1
    # Total words: 3^6 = 729
    total_words = 729
    powers = [1, 3, 9, 27, 81, 243]
    
    # Precompute ball masks for all 729 words
    # ball_mask[i] is an integer with bits set for all words within Hamming distance 1 of i
    ball_mask = [0] * total_words
    for i in range(total_words):
        # Get digits
        digits = [(i // p) % 3 for p in powers]
        mask = 1 << i
        for pos in range(6):
            orig = digits[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                # Calculate new index
                # idx = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4 + 243*d5
                # delta = (new_val - orig) * powers[pos]
                new_idx = i + (new_val - orig) * powers[pos]
                mask |= (1 << new_idx)
        ball_mask[i] = mask

    # Initial Solution: Tetracode product
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Product with all 3^2 = 9 words of length 2
    initial_codewords = []
    for h in tetracode:
        for tail in [(c, d) for c in range(3) for d in range(3)]:
            w = h + list(tail)
            # Convert to index
            idx = 0
            for k, v in enumerate(w):
                idx += v * powers[k]
            initial_codewords.append(idx)
    
    codewords = initial_codewords[:]
    in_set = set(codewords)
    
    # Helper to compute union of balls
    def get_coverage(cw_list):
        cov = 0
        for c in cw_list:
            cov |= ball_mask[c]
        return cov

    # Helper to check redundancy
    def is_redundant(cw, cw_list, in_set_local):
        # Compute union of balls of all others
        u = 0
        for c in cw_list:
            if c != cw:
                u |= ball_mask[c]
        # Check if ball of cw is subset of u
        return (ball_mask[cw] & ~u) == 0

    # Phase 1: Greedy Removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate in reverse to handle index changes if we pop, but we use list
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                return [list(((c // powers[k]) % 3 for k in range(6))) for c in codewords]
            if is_redundant(codewords[i], codewords, in_set):
                cw = codewords[i]
                codewords.pop(i)
                in_set.discard(cw)
                improved = True
                break # Restart scan
    
    # Phase 2: Local Search / Simulated Annealing
    # We will perform random swaps to escape local minima
    # Move: Remove one codeword, add one new codeword that covers the "critical" words
    # This maintains size but changes configuration
    
    random.seed(42)
    np.random.seed(42)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    best_size = len(codewords)
    best_codewords = codewords[:]
    
    while time.time() < deadline - 5:
        # Check for redundant words again (might have appeared after swaps)
        removed = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(codewords[i], codewords, in_set):
                cw = codewords[i]
                codewords.pop(i)
                in_set.discard(cw)
                removed = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = codewords[:]
                break
        
        if removed:
            T = max(T * cooling, T_min)
            continue
            
        # No redundant words. Try a swap.
        # Pick a random codeword to remove
        i = random.randint(0, len(codewords) - 1)
        cw_remove = codewords[i]
        
        # Compute union of others
        u_others = 0
        for c in codewords:
            if c != cw_remove:
                u_others |= ball_mask[c]
        
        # Critical words: in ball(cw_remove) but not in u_others
        critical = ball_mask[cw_remove] & ~u_others
        
        if critical == 0:
            # Should have been caught by redundancy check, but just in case
            codewords.pop(i)
            in_set.discard(cw_remove)
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = codewords[:]
            continue
            
        # Find candidates: words whose ball covers 'critical'
        # To speed up, we can precompute for each 'critical' pattern? No, critical varies.
        # Iterate through all 729 words and check if ball_mask[j] covers critical
        candidates = []
        for j in range(total_words):
            if j in in_set:
                continue
            if (ball_mask[j] & critical) == critical:
                candidates.append(j)
                
        if not candidates:
            # No valid swap found for this removal. Try another removal or skip.
            # To avoid getting stuck, we could try a "2-swap" or just restart, 
            # but simple random selection of removal word usually finds a path.
            # Let's try a few more random removals in this iteration if time permits
            # For now, just cool down and try next time
            T = max(T * cooling, T_min)
            continue
            
        # Pick a candidate
        # Heuristic: pick one that minimizes "waste" (covers fewest extra words) or just random
        # Random is good for SA exploration
        cw_add = random.choice(candidates)
        
        # Perform swap
        codewords[i] = cw_add
        in_set.discard(cw_remove)
        in_set.add(cw_add)
        
        # Since size didn't change, best_size doesn't update unless we find a smaller one later
        T = max(T * cooling, T_min)
        
    # Final cleanup: greedy removal pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(codewords[i], codewords, in_set):
                cw = codewords[i]
                codewords.pop(i)
                in_set.discard(cw)
                improved = True
                break
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = codewords[:]
        
    # Return the best found
    result_indices = best_codewords
    
    # Convert indices back to lists of ints
    result = []
    for idx in result_indices:
        w = [0] * 6
        for k in range(6):
            w[k] = (idx // powers[k]) % 3
        result.append(w)
        
    return result

# EVOLVE-BLOCK-END
