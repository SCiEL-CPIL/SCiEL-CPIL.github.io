# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(q, n, radius):
    """
    Optimized greedy set-cover algorithm with high-volume random restarts.
    Designed for small spaces (q^n < 10000) where bitmask operations are efficient.
    Includes a post-processing reduction phase to remove redundant codewords.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words > 5000:
        # Fallback for larger spaces, though this task is small
        return _construct_greedy_simple(q, n, radius)
        
    return _construct_greedy_fast(q, n, radius)

def _construct_greedy_fast(q, n, radius):
    """
    Fast greedy set-cover using precomputed bitmask balls and arithmetic neighbor generation.
    Includes a multi-pass reduction phase to minimize the set size.
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
                # For each other value v in 0..q-1
                for v in range(q):
                    if v != digit:
                        j = i + (v - digit) * step
                        mask |= 1 << j
            ball_masks[i] = mask
    else:
        # Generic fallback for radius > 1
        # This is slower but handles other cases
        def idx_to_word(idx):
            word = [0] * n
            for k in range(n-1, -1, -1):
                word[k] = idx % q
                idx //= q
            return word
        
        for i in range(total_words):
            mask = 1 << i
            center = idx_to_word(i)
            if radius == 2:
                # Distance 1
                for p in range(n):
                    step = q_powers[p]
                    digit = (i // step) % q
                    for v in range(q):
                        if v != digit:
                            j = i + (v - digit) * step
                            mask |= 1 << j
                # Distance 2
                for p1 in range(n):
                    for p2 in range(p1+1, n):
                        step1 = q_powers[p1]
                        step2 = q_powers[p2]
                        d1 = (i // step1) % q
                        d2 = (i // step2) % q
                        for v1 in range(q):
                            if v1 != d1:
                                for v2 in range(q):
                                    if v2 != d2:
                                        j = i + (v1 - d1) * step1 + (v2 - d2) * step2
                                        mask |= 1 << j
            ball_masks[i] = mask

    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    # Determine number of restarts based on time budget
    start_time = time.time()
    max_time = 110.0 # Leave 10s for overhead
    
    restart_count = 0
    rng = np.random.RandomState(42)
    
    # Helper to convert idx to word
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for k in range(n-1, -1, -1):
            word[k] = temp % q
            temp //= q
        return word

    indices = list(range(total_words))
    
    while True:
        # Generate random permutation for tie-breaking
        rng.shuffle(indices)
        
        uncovered = all_mask
        codeword_indices = []
        
        while uncovered != 0:
            max_score = 0
            best_idx = -1
            
            for idx in indices:
                score = (ball_masks[idx] & uncovered).bit_count()
                if score > max_score:
                    max_score = score
                    best_idx = idx
                    
            if max_score == 0:
                break
                
            codeword_indices.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
        # Reduction Phase: Remove redundant codewords
        if len(codeword_indices) > 0:
            reduced_indices = _reduce_codewords(codeword_indices, ball_masks, total_words)
            
            # Convert to words
            codewords = [idx_to_word(idx) for idx in reduced_indices]
            
            if len(codewords) < best_len:
                best_len = len(codewords)
                best_codewords = codewords
                
                # Early termination if we hit known optimal
                if best_len == 73:
                    break
                    
        restart_count += 1
        
        # Check time
        if time.time() - start_time > max_time:
            break
            
        # Safety cap on restarts
        if restart_count > 20000:
            break
            
    return best_codewords

def _reduce_codewords(indices, ball_masks, total_words):
    """
    Multi-pass greedy reduction to remove redundant codewords.
    """
    active = list(indices)
    if not active:
        return active

    # Calculate initial coverage counts
    coverage_count = [0] * total_words
    for idx in active:
        mask = ball_masks[idx]
        m = mask
        while m:
            lsb = m & -m
            w_idx = lsb.bit_length() - 1
            coverage_count[w_idx] += 1
            m ^= lsb

    changed = True
    while changed:
        changed = False
        new_active = []
        for idx in active:
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
                # Remove it and update coverage counts
                m = mask
                while m:
                    lsb = m & -m
                    w_idx = lsb.bit_length() - 1
                    coverage_count[w_idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_active.append(idx)
        active = new_active
        
    return active

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
