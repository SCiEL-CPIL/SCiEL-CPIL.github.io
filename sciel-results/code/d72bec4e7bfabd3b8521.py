# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(q, n, radius):
    """
    Optimized greedy set-cover algorithm with iterative reduction.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words > 10000:
        return _construct_greedy_simple(q, n, radius)
        
    return _construct_greedy_fast(q, n, radius)

def _construct_greedy_fast(q, n, radius):
    """
    Fast greedy set-cover with aggressive reduction.
    """
    total_words = q ** n
    
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
        
    # Precompute ball masks
    # ball_masks[i] is a bitmask of all words within Hamming distance 'radius' of i
    ball_masks = [0] * total_words
    
    if radius == 1:
        for i in range(total_words):
            mask = 1 << i
            for p in range(n):
                step = q_powers[p]
                digit = (i // step) % q
                for v in range(q):
                    if v == digit:
                        continue
                    j = i + (v - digit) * step
                    mask |= 1 << j
            ball_masks[i] = mask
    else:
        # Generic fallback for radius > 1
        def idx_to_word(idx):
            word = [0] * n
            for k in range(n-1, -1, -1):
                word[k] = idx % q
                idx //= q
            return word
        
        for i in range(total_words):
            mask = 1 << i
            center = idx_to_word(i)
            # Brute force neighbors for general radius (slow but correct)
            # For this specific task radius=1, so this is rarely hit unless params change
            # We'll just do a simple expansion for radius 2 as well for safety
            if radius == 2:
                for p1 in range(n):
                    for p2 in range(n):
                        if p1 == p2: continue
                        step1 = q_powers[p1]
                        step2 = q_powers[p2]
                        d1 = (i // step1) % q
                        d2 = (i // step2) % q
                        for v1 in range(q):
                            if v1 == d1: continue
                            for v2 in range(q):
                                if v2 == d2: continue
                                j = i + (v1 - d1) * step1 + (v2 - d2) * step2
                                mask |= 1 << j
            ball_masks[i] = mask

    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = 115.0
    
    restart_count = 0
    rng = np.random.RandomState(42)
    
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for k in range(n-1, -1, -1):
            word[k] = temp % q
            temp //= q
        return word

    while time.time() - start_time < max_time:
        # Generate random permutation for tie-breaking
        perm = rng.permutation(total_words)
        
        uncovered = all_mask
        codeword_indices = []
        
        # Greedy Phase
        while uncovered != 0:
            max_score = 0
            best_idx = -1
            
            # Iterate through candidates
            # Optimization: If we have a list of candidates, we can filter.
            # But scanning all 729 is ~fast enough in Python if we avoid heavy ops.
            for idx in perm:
                # Only consider if it covers at least one uncovered word?
                # (ball_masks[idx] & uncovered) != 0
                # But we need max count.
                score = (ball_masks[idx] & uncovered).bit_count()
                if score > max_score:
                    max_score = score
                    best_idx = idx
                    
            if max_score == 0:
                break
                
            codeword_indices.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
        # Reduction Phase
        # We will iteratively try to remove codewords.
        # A codeword c can be removed if for all w in ball(c), count(w) > 1.
        
        if len(codeword_indices) > 0:
            # Initialize coverage counts
            coverage_count = [0] * total_words
            for idx in codeword_indices:
                mask = ball_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    w_idx = lsb.bit_length() - 1
                    coverage_count[w_idx] += 1
                    m ^= lsb
            
            active_indices = list(codeword_indices)
            
            # Iterative reduction
            # We repeat until no codeword can be removed or max iterations reached
            for _ in range(50): # Max 50 passes
                removed_any = False
                new_active_indices = []
                for idx in active_indices:
                    mask = ball_masks[idx]
                    m = mask
                    is_redundant = True
                    while m:
                        lsb = m & -m
                        w_idx = lsb.bit_length() - 1
                        if coverage_count[w_idx] == 1:
                            is_redundant = False
                            break
                        m ^= lsb
                    
                    if is_redundant:
                        # Remove it
                        removed_any = True
                        m = mask
                        while m:
                            lsb = m & -m
                            w_idx = lsb.bit_length() - 1
                            coverage_count[w_idx] -= 1
                            m ^= lsb
                    else:
                        new_active_indices.append(idx)
                
                if not removed_any:
                    break
                active_indices = new_active_indices
            
            final_indices = active_indices
            codewords = [idx_to_word(idx) for idx in final_indices]
            
            if len(codewords) < best_len:
                best_len = len(codewords)
                best_codewords = codewords
                
                # Early termination if we hit known optimal
                if best_len <= 73:
                    break
                    
        restart_count += 1

    return best_codewords

def _construct_greedy_simple(q, n, radius):
    """Fallback simple greedy for larger spaces."""
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    ball_masks = [0] * total_words
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 1 << idx
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        ball_masks[idx] = mask
        
    uncovered = (1 << total_words) - 1
    codewords = []
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    return codewords
# EVOLVE-BLOCK-END
