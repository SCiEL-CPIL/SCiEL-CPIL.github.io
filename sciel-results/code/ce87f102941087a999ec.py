import numpy as np
import time
import random

N_WORDS = 3**8

# Precompute masks
# masks[i] is an array of 25 indices: i and its 24 neighbors
_masks = None
_idx_to_word = None

def _init():
    global _masks, _idx_to_word
    if _masks is not None:
        return
    
    # Precompute word representations
    _idx_to_word = []
    for i in range(N_WORDS):
        digits = [0]*8
        tmp = i
        for k in range(8):
            digits[k] = tmp % 3
            tmp //= 3
        _idx_to_word.append(digits)
        
    # Precompute masks
    _masks = np.zeros((N_WORDS, 25), dtype=np.int32)
    for i in range(N_WORDS):
        _masks[i, 0] = i
        # Generate neighbors
        digits = _idx_to_word[i]
        neigh_idx = 1
        for k in range(8):
            orig = digits[k]
            for new_val in range(3):
                if new_val != orig:
                    # Calculate new index
                    # i - orig * 3^k + new_val * 3^k
                    new_i = i - orig * (3**k) + new_val * (3**k)
                    _masks[i, neigh_idx] = new_i
                    neigh_idx += 1
        # neigh_idx should be 25

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected)
        # Simple greedy without optimization
        return _fallback_construct(q, n, radius)
        
    _init()
    masks = _masks
    idx_to_word = _idx_to_word
    
    best_code = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115
    
    while time.time() < deadline:
        uncovered = np.ones(N_WORDS, dtype=np.uint8)
        cover_counts = np.zeros(N_WORDS, dtype=np.int32)
        scores = np.full(N_WORDS, 25, dtype=np.int32)
        
        code = []
        uncovered_count = N_WORDS
        
        # Greedy Phase with incremental score update
        while uncovered_count > 0:
            max_score = scores.max()
            candidates = np.where(scores == max_score)[0]
            c = random.choice(candidates.tolist())
            
            code.append(c)
            
            covered_indices = masks[c]
            
            # Update cover counts
            cover_counts[covered_indices] += 1
            
            # Update uncovered and scores
            for w in covered_indices:
                if uncovered[w] == 1:
                    uncovered[w] = 0
                    uncovered_count -= 1
                    # Update scores for all codewords covering w
                    # Since masks is symmetric, codewords covering w are masks[w]
                    scores[masks[w]] -= 1
        
        # Deletion Phase
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find all removable codewords
            removable = []
            for i, c in enumerate(code):
                # c is removable if all words it covers have count > 1
                if np.all(cover_counts[masks[c]] > 1):
                    removable.append(i)
                    
            if not removable:
                break
                
            idx_to_remove = random.choice(removable)
            c = code[idx_to_remove]
            cover_counts[masks[c]] -= 1
            code.pop(idx_to_remove)
            changed = True
            
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
    if best_code is None:
        return []
        
    return [idx_to_word[idx] for idx in best_code]

def _fallback_construct(q, n, radius):
    # Basic greedy for non-standard params
    # Not optimized, just to avoid crash
    all_words = []
    import itertools
    for w in itertools.product(range(q), repeat=n):
        all_words.append(list(w))
        
    # Simple greedy
    uncovered = set(range(len(all_words)))
    code = []
    while uncovered:
        best_c = None
        best_count = -1
        for c in all_words:
            # Count how many uncovered words are covered by c
            # This is O(N^2 * n) which is too slow for large N, but fallback
            # For small N it's ok.
            count = 0
            for u_idx in uncovered:
                u = all_words[u_idx]
                dist = sum(1 for i in range(n) if u[i] != c[i])
                if dist <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_c = c
        code.append(best_c)
        # Remove covered
        to_remove = []
        for u_idx in uncovered:
            u = all_words[u_idx]
            dist = sum(1 for i in range(n) if u[i] != best_c[i])
            if dist <= radius:
                to_remove.append(u_idx)
        for idx in to_remove:
            uncovered.remove(idx)
    return code
