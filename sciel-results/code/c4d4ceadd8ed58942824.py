# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Multi-start greedy set-cover algorithm to find a small covering code.
    Uses NumPy vectorized operations for fast greedy selection and multiple
    random tie-breaking orders to escape local optima.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words <= 2**20:
        return _construct_greedy_multi_start(q, n, radius)
    else:
        # Fallback for large spaces (not expected for this task, but safe)
        return _construct_greedy_simple(q, n, radius)


def _construct_greedy_multi_start(q, n, radius):
    """
    Optimized greedy set-cover with multiple restarts for small spaces.
    Uses NumPy for vectorized score calculation.
    """
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

    # Precompute ball masks as a NumPy boolean matrix B[i, j] = 1 if j is in ball(i)
    B = np.zeros((total_words, total_words), dtype=bool)
    
    for idx in range(total_words):
        center = idx_to_word(idx)
        B[idx, idx] = True
        
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        B[idx, nidx] = True
        
        if radius >= 2:
            # Add distance 2 neighbors
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == center[pos1]: continue
                        for val2 in range(q):
                            if val2 == center[pos2]: continue
                            neighbor = center.copy()
                            neighbor[pos1] = val1
                            neighbor[pos2] = val2
                            nidx = word_to_idx(neighbor)
                            B[idx, nidx] = True

    best_codewords = None
    best_len = float('inf')
    
    # Increase restarts significantly due to speedup
    num_restarts = 1000
    
    for seed in range(num_restarts):
        # Generate a random permutation of indices for tie-breaking
        perm = np.random.RandomState(seed).permutation(total_words)
        # Precompute rank for each index in the permutation for O(1) tie-breaking
        rank = np.empty(total_words, dtype=np.int32)
        rank[perm] = np.arange(total_words)
        
        uncovered = np.ones(total_words, dtype=bool)
        codewords = []
        
        while np.any(uncovered):
            # Calculate scores: number of newly covered words for each candidate
            # B & uncovered[:, np.newaxis] broadcasts uncovered to (N, N)
            # Sum along axis 1 gives count of True values in each row
            scores = np.sum(B & uncovered[:, np.newaxis], axis=1)
            
            max_score = np.max(scores)
            if max_score == 0:
                break # Should not happen
                
            # Find all candidates with max score
            candidates = np.where(scores == max_score)[0]
            
            # Select the candidate with the smallest rank (earliest in random permutation)
            cand_ranks = rank[candidates]
            best_local_idx = np.argmin(cand_ranks)
            best_idx = candidates[best_local_idx]
            
            codewords.append(idx_to_word(best_idx))
            # Update uncovered
            uncovered = uncovered & ~B[best_idx]
            
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords
            
            # Early termination if we hit a very good bound (e.g., known optimal)
            # For K_3(6,1), optimal is 73.
            if best_len == 73:
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
