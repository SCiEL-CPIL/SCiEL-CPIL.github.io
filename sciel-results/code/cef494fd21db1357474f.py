# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(q, n, radius):
    """
    Optimized greedy set-cover algorithm using NumPy for fast scoring and iterative reduction.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words > 5000:
        return _construct_greedy_simple(q, n, radius)
        
    return _construct_greedy_numpy(q, n, radius)

def _construct_greedy_numpy(q, n, radius):
    total_words = q ** n
    
    # Precompute balls as a boolean NumPy array
    # balls[i, j] is True if word j is within radius of word i
    balls = np.zeros((total_words, total_words), dtype=bool)
    
    # Generate all words
    words = np.array(np.unravel_index(np.arange(total_words), (q,)*n), dtype=int).T
    # words shape: (total_words, n)
    
    # Compute distances or just set balls directly
    # For radius 1:
    if radius == 1:
        for i in range(total_words):
            balls[i, i] = True
            for p in range(n):
                # Get neighbors by changing digit p
                for v in range(q):
                    if v == words[i, p]:
                        continue
                    # Create neighbor
                    # Instead of creating full array, we can calculate index
                    # But we already have words array.
                    # We need to map neighbor back to index.
                    # This is slow in Python loop.
                    # Let's use the index arithmetic method to fill balls, or just compute distance.
                    pass
        
        # Faster way: compute distance matrix?
        # dist[i,j] = sum(words[i] != words[j])
        # This is O(N^2 * n). 729^2 * 6 ~ 3.2M ops. Very fast in NumPy.
        diff = words[:, np.newaxis, :] != words[np.newaxis, :, :]
        dist = diff.sum(axis=2)
        balls = dist <= radius
    else:
        # Generic fallback for radius > 1 using distance matrix
        diff = words[:, np.newaxis, :] != words[np.newaxis, :, :]
        dist = diff.sum(axis=2)
        balls = dist <= radius

    all_covered = np.ones(total_words, dtype=bool)
    best_codewords = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = 110.0
    
    rng = np.random.RandomState(42)
    
    restart_count = 0
    
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for k in range(n-1, -1, -1):
            word[k] = temp % q
            temp //= q
        return word

    while True:
        uncovered = np.ones(total_words, dtype=bool)
        codeword_indices = []
        
        while uncovered.any():
            # Score each candidate: number of uncovered words it covers
            # counts[i] = sum(balls[i] & uncovered)
            counts = balls.dot(uncovered.astype(np.int8))
            
            # Add noise to break ties randomly
            noise = rng.random(total_words)
            counts += noise
            
            best_idx = np.argmax(counts)
            
            if counts[best_idx] <= 0: # Should not happen if uncovered.any()
                break
                
            codeword_indices.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        # Reduction Phase: Iteratively remove redundant codewords
        # coverage_counts[w] = number of codewords covering w
        coverage_counts = np.zeros(total_words, dtype=int)
        for idx in codeword_indices:
            coverage_counts += balls[idx].astype(int)
            
        active_indices = list(codeword_indices)
        
        # Iterate until no more removals
        changed = True
        while changed:
            changed = False
            # Iterate in reverse or random order? 
            # Let's iterate in the current order.
            for i in range(len(active_indices)):
                idx = active_indices[i]
                # Check if any word in ball(idx) has coverage 1
                # If min coverage in ball is > 1, it's redundant
                if coverage_counts[balls[idx]].min() > 1:
                    # Remove idx
                    coverage_counts -= balls[idx].astype(int)
                    active_indices[i] = -1
                    changed = True
            # Filter out removed
            active_indices = [idx for idx in active_indices if idx != -1]
            
        if len(active_indices) < best_len:
            best_len = len(active_indices)
            best_codewords = [idx_to_word(idx) for idx in active_indices]
            
            if best_len == 73:
                break
                
        restart_count += 1
        
        if time.time() - start_time > max_time:
            break
        if restart_count > 500: # NumPy is fast, but we don't need infinite
            break
            
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
