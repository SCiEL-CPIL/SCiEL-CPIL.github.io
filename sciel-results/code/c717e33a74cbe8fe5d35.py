import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = q ** n
    
    # Precompute spheres
    spheres = []
    for i in range(N):
        w = [0] * n
        temp = i
        for j in range(n):
            w[j] = temp % q
            temp //= q
        
        sphere = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                w[pos] = new_val
                new_idx = 0
                mult = 1
                for j in range(n):
                    new_idx += w[j] * mult
                    mult *= q
                sphere.append(new_idx)
        spheres.append(np.array(sphere))
        
    def word_to_idx(word):
        idx = 0
        mult = 1
        for j in range(n):
            idx += word[j] * mult
            mult *= q
        return idx
        
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for j in range(n):
            w[j] = temp % q
            temp //= q
        return w
        
    def reduce_code(codewords_idx):
        # Redundancy Removal
        if time.time() > deadline - 5:
            return codewords_idx
            
        coverage_count = np.zeros(N, dtype=int)
        for c in codewords_idx:
            coverage_count[spheres[c]] += 1
            
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_codewords = []
            for c in codewords_idx:
                # If any word in sphere[c] has count 1, c is needed
                # If all words in sphere[c] have count > 1, c is redundant
                if not np.any(coverage_count[spheres[c]] == 1):
                    coverage_count[spheres[c]] -= 1
                    changed = True
                else:
                    new_codewords.append(c)
            codewords_idx = new_codewords
        return codewords_idx
        
    best_codewords = None
    best_len = N
    
    # Strategy 1: Direct Sum + Reduction
    if time.time() < deadline - 5:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        ds_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        ds_idx = [word_to_idx(w) for w in ds_words]
        reduced_ds = reduce_code(ds_idx)
        if best_codewords is None or len(reduced_ds) < best_len:
            best_len = len(reduced_ds)
            best_codewords = reduced_ds.copy()
            
    # Strategy 2: Greedy Set Cover + Reduction
    # Run a few times with random tie-breaking
    trials = 0
    while time.time() < deadline - 10 and trials < 20:
        trials += 1
        uncovered = np.ones(N, dtype=bool)
        codewords = []
        
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            
            best_c = -1
            max_covered = -1
            candidates = []
            
            for c in range(N):
                covered_count = np.sum(uncovered[spheres[c]])
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_c = c
                    candidates = [c]
                elif covered_count == max_covered:
                    candidates.append(c)
            
            if best_c == -1:
                break
                
            c = random.choice(candidates)
            codewords.append(c)
            uncovered[spheres[c]] = False
            
        reduced_greedy = reduce_code(codewords)
        if best_codewords is None or len(reduced_greedy) < best_len:
            best_len = len(reduced_greedy)
            best_codewords = reduced_greedy.copy()
            
    if best_codewords is None:
        best_codewords = list(range(N))
        
    result = [idx_to_word(c) for c in best_codewords]
    return result
