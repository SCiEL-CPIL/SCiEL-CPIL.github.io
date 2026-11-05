import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    return _greedy_general(q, n, radius)

def _solve_ternary_8_1():
    start_time = time.time()
    deadline = start_time + 115.0
    
    q = 3
    n = 8
    total = q ** n  # 6561
    
    # Generate all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (Hamming distance <= 1) for each word
    # neighbors[i] is a list of indices of words within dist 1 of word i
    neighbors = []
    for i, w in enumerate(all_words):
        nbrs = [i]
        w_list = list(w)
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        neighbors.append(nbrs)
        
    # Convert to numpy array for faster access if needed, but list of lists is fine for small k
    # neighbors_np = np.array(neighbors) # shape (6561, 17)
    
    # Greedy Construction
    uncovered = np.ones(total, dtype=bool)
    code_indices = []
    
    while np.any(uncovered) and time.time() < deadline:
        # Calculate gain for each candidate
        # gain[i] = number of uncovered words in neighbors[i]
        # Vectorized approach:
        # We can't easily vectorize the "sum of uncovered at indices" without advanced indexing or loops.
        # However, 6561 * 17 is small.
        
        gains = np.zeros(total, dtype=np.int32)
        # Using a loop is safer for correctness and likely fast enough (< 0.1s per step)
        # But let's try to be efficient.
        # For each i, sum(uncovered[neighbors[i]])
        
        # Optimization: Only consider candidates that are currently "active" or just all of them?
        # All of them is fine.
        
        # This loop is the bottleneck. 6561 iterations.
        # We can speed this up by noting that if a word is uncovered, it contributes to the gain of its neighbors.
        # Let's invert the logic:
        # gain[i] = 0
        # for each uncovered word u:
        #     for each neighbor v of u:
        #         gain[v] += 1
        # This is: number of uncovered words * 17.
        # Initially 6561 * 17. Later, much less.
        # This is likely faster than iterating all 6561 candidates every time.
        
        gains = np.zeros(total, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        
        if len(uncovered_indices) == 0:
            break
            
        # For each uncovered word, increment gain of its neighbors
        # neighbors is a list of lists.
        # We can precompute an "inverse neighbors" or just use the fact that dist is symmetric.
        # If u is uncovered, it is in neighbors[v] iff v is in neighbors[u].
        # So we just iterate over uncovered words and add 1 to gain of all their neighbors.
        
        for u in uncovered_indices:
            for v in neighbors[u]:
                gains[v] += 1
                
        # Find candidate with max gain
        best_c = np.argmax(gains)
        best_gain = gains[best_c]
        
        if best_gain == 0:
            break # Should not happen if uncovered is not empty
            
        # Add best_c to code
        code_indices.append(best_c)
        
        # Update uncovered
        for v in neighbors[best_c]:
            uncovered[v] = False
            
    # Local Search: Redundancy Removal
    # Maintain cover_count: how many codewords cover each word
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_indices:
        for v in neighbors[c]:
            cover_count[v] += 1
            
    # Iteratively remove redundant codewords
    # A codeword c is redundant if for all v in neighbors[c], cover_count[v] > 1
    # We iterate until no more removals.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate over a copy of the list to allow removal
        # We need to be careful with indices.
        # Let's use a list of indices.
        
        # To make removal efficient, we can just iterate and mark for removal.
        to_remove = []
        
        for c in code_indices:
            # Check if c is redundant
            redundant = True
            for v in neighbors[c]:
                if cover_count[v] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(c)
                
        if to_remove:
            # Remove all redundant codewords
            for c in to_remove:
                code_indices.remove(c)
                for v in neighbors[c]:
                    cover_count[v] -= 1
            changed = True
        else:
            # No redundancy found in this pass.
            # We could try swaps here, but redundancy removal is the primary driver for size reduction after greedy.
            # Greedy tends to over-cover.
            break
            
    # Convert back to words
    codewords = [list(all_words[c]) for c in code_indices]
    codewords.sort()
    return codewords

def _greedy_general(q, n, radius):
    # Fallback for other parameters
    # Simple greedy using distance calculation
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    total = len(all_words)
    uncovered = [True] * total
    code = []
    
    for i in range(total):
        if not uncovered[i]:
            continue
        # Find best codeword covering i and most others
        best_c = -1
        best_count = -1
        for c in range(total):
            # Calculate distance
            dist = sum(1 for j in range(n) if all_words[i][j] != all_words[c][j])
            if dist <= radius:
                # Count how many uncovered words c covers
                count = 0
                for u in range(total):
                    if uncovered[u]:
                        du = sum(1 for j in range(n) if all_words[u][j] != all_words[c][j])
                        if du <= radius:
                            count += 1
                if count > best_count:
                    best_count = count
                    best_c = c
        if best_c != -1:
            code.append(all_words[best_c])
            for u in range(total):
                du = sum(1 for j in range(n) if all_words[u][j] != all_words[best_c][j])
                if du <= radius:
                    uncovered[u] = False
    return code
