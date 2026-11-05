import itertools
import numpy as np
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    # Specific for q=3, n=8, r=1
    total_words = 3 ** 8
    
    # Precompute idx <-> word and neighborhoods
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    neighborhoods = []
    for i in range(total_words):
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(3):
                if val != orig:
                    new_w = w[:]
                    new_w[pos] = val
                    neigh.append(word_to_idx(new_w))
        neighborhoods.append(neigh)

    def run_greedy(seed):
        random.seed(seed)
        uncovered = np.ones(total_words, dtype=np.int32)
        code_indices = []
        
        while np.any(uncovered > 0):
            best_gain = -1
            best_candidates = []
            
            # To speed up, we can check all, but break early if we find gain == 17?
            # Max gain is 17.
            
            for i in range(total_words):
                # Optimization: If uncovered[i] == 0 and all neighbors are covered, gain is 0.
                # But checking is O(17).
                
                gain = 0
                for idx in neighborhoods[i]:
                    if uncovered[idx] > 0:
                        gain += 1
                        if gain + (17 - gain) <= best_gain: # Even if all remaining are uncovered, we can't beat best?
                             # This logic is flawed because we don't know remaining.
                             pass
                
                # Simpler: just calculate full gain.
                # 55M ops is fine.
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
                
            # Random tie-breaking
            chosen = random.choice(best_candidates)
            code_indices.append(chosen)
            
            for idx in neighborhoods[chosen]:
                uncovered[idx] -= 1
                
        return code_indices

    # Run greedy multiple times
    best_code_indices = None
    min_size = float('inf')
    
    # 5 restarts
    for seed in range(5):
        code_indices = run_greedy(seed)
        if len(code_indices) < min_size:
            min_size = len(code_indices)
            best_code_indices = code_indices
            
    # Convert to list of lists for peeling
    code = [idx_to_word(i) for i in best_code_indices]
    
    # Peeling Phase
    # Convert to numpy array for efficient ops?
    # We need to maintain coverage counts.
    
    m = len(code)
    if m == 0:
        return code
        
    # Map word to index in the code list?
    # No, we need global index (0..6560) for coverage.
    
    # Precompute neighborhoods for the *selected* codewords only?
    # We already have `neighborhoods` for all 6561 words.
    
    # Coverage array
    coverage = np.zeros(total_words, dtype=np.int32)
    
    # List of active codeword indices (global indices 0..6560)
    active_indices = list(best_code_indices)
    
    # Initialize coverage
    for idx in active_indices:
        for n_idx in neighborhoods[idx]:
            coverage[n_idx] += 1
            
    changed = True
    while changed:
        changed = False
        best_remove_idx = -1
        best_redundancy = 0
        
        # Iterate through active indices
        # To avoid index shifting issues, we can use a list and pop.
        
        for i in range(len(active_indices)):
            c_idx = active_indices[i]
            neigh = neighborhoods[c_idx]
            
            # Check if removable
            removable = True
            redundancy = 0
            for n_idx in neigh:
                if coverage[n_idx] <= 1:
                    removable = False
                    break
                # Count how many are covered by others (coverage > 1)
                if coverage[n_idx] > 1:
                    redundancy += 1
                    
            if removable:
                if redundancy > best_redundancy:
                    best_redundancy = redundancy
                    best_remove_idx = i
                    
        if best_remove_idx != -1:
            c_idx = active_indices[best_remove_idx]
            # Remove from active_indices
            active_indices.pop(best_remove_idx)
            # Update coverage
            for n_idx in neighborhoods[c_idx]:
                coverage[n_idx] -= 1
            changed = True
            
    return [idx_to_word(i) for i in active_indices]
