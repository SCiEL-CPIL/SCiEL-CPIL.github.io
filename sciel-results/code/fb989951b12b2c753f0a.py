import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w
    
    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute sphere neighbors for each word
    # sphere_neighbors[i] = list of indices in sphere of word i (including i)
    sphere_neighbors = []
    for i in range(total_words):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                j = word_to_idx(new_w)
                neighbors.append(j)
        sphere_neighbors.append(neighbors)
    
    sphere_size = len(sphere_neighbors[0])
    
    # Precompute sphere as boolean matrix for pruning
    sphere_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        for j in sphere_neighbors[i]:
            sphere_matrix[i, j] = 1
    
    # Precompute sphere columns for fast gain updates
    sphere_cols = []
    for j in range(total_words):
        col = np.zeros(total_words, dtype=np.int32)
        for i in range(total_words):
            if j in sphere_neighbors[i]:
                col[i] = 1
        sphere_cols.append(col)
    
    # Initial solution: tetracode construction (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    best_codewords = initial_codewords
    best_size = len(best_codewords)
    
    start_time = time.time()
    deadline = start_time + 115  # 5 second margin
    
    def prune_codewords(cw_indices):
        """Remove redundant codewords from the list."""
        if len(cw_indices) == 0:
            return cw_indices
        
        cw_indices = cw_indices[:]
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in cw_indices:
            for j in sphere_neighbors[cw]:
                counts[j] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(cw_indices)):
                cw = cw_indices[i]
                can_remove = True
                for j in sphere_neighbors[cw]:
                    if counts[j] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for j in sphere_neighbors[cw]:
                        counts[j] -= 1
                    cw_indices.pop(i)
                    improved = True
                    break
        
        return cw_indices
    
    # Greedy set cover with incremental updates and randomization
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        uncovered = np.ones(total_words, dtype=np.uint8)
        gains = np.full(total_words, sphere_size, dtype=np.int32)
        current_codewords = []
        
        while True:
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[random.randint(0, len(candidates) - 1)]
            current_codewords.append(chosen_idx)
            
            # Update uncovered and gains incrementally
            newly_covered = sphere_matrix[chosen_idx] & uncovered
            uncovered = uncovered & (~sphere_matrix[chosen_idx])
            
            for j in np.where(newly_covered)[0]:
                gains -= sphere_cols[j]
            
            if uncovered.sum() == 0:
                break
        
        # Prune redundant codewords
        if len(current_codewords) < best_size:
            pruned = prune_codewords(current_codewords)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_codewords = [all_words[i] for i in pruned]
    
    return best_codewords
