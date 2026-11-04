import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Generate all words and indices
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors
    neighbors = []
    for i, w in enumerate(words):
        neigh = set([i])
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neigh.add(idx)
        neighbors.append(list(neigh))
        
    # Create a boolean matrix M where M[i, j] is 1 if word j covers word i
    M = np.zeros((N, N), dtype=np.uint8)
    for j in range(N):
        for i in neighbors[j]:
            M[i, j] = 1
            
    # 3. Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    while np.any(uncovered):
        new_cov = M.T @ uncovered.astype(np.uint8)
        best_idx = np.argmax(new_cov)
        if new_cov[best_idx] == 0:
            break
            
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        if time.time() > deadline - 30:
            break

    # 4. Local Search
    counts = np.zeros(N, dtype=np.int16)
    for idx in code:
        counts[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Deletion Pass
        to_remove = []
        for c in code:
            if np.min(counts[neighbors[c]]) > 1:
                to_remove.append(c)
                
        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                counts[neighbors[c]] -= 1
            
        # Swap Pass
        random.shuffle(code)
        code_set = set(code)
        
        for c in code:
            if time.time() > deadline - 10:
                break
                
            # Critical points for c
            crit_mask = (counts == 1) & M[:, c].astype(bool)
            crit_points = np.where(crit_mask)[0]
            
            if len(crit_points) == 0:
                continue
                
            best_w = -1
            best_score = -1000
            
            # Precompute sets for scoring
            # N_c = set(neighbors[c])
            # N_w = set(neighbors[w])
            
            for w in range(N):
                if w in code_set:
                    continue
                
                # Check validity: crit_points subset of neighbors[w]
                # M[crit_points, w] must be all 1
                if not np.all(M[crit_points, w]):
                    continue
                
                # Calculate score
                # Points in N(w) not in N(c) with count 1
                # Points in N(c) not in N(w) with count 2
                
                # Using M matrix:
                # in_w_not_c = M[:, w].astype(bool) & ~M[:, c].astype(bool)
                # in_c_not_w = M[:, c].astype(bool) & ~M[:, w].astype(bool)
                
                # This is expensive inside the loop.
                # Optimization:
                # We can precompute M_bool = M.astype(bool)
                # But M is uint8.
                
                # Let's just use the lists for small N
                n_w = set(neighbors[w])
                n_c = set(neighbors[c])
                
                diff_w = n_w - n_c
                diff_c = n_c - n_w
                
                score = 0
                for i in diff_w:
                    if counts[i] == 1:
                        score += 1
                for i in diff_c:
                    if counts[i] == 2:
                        score -= 1
                        
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1 and best_score > 0:
                code.remove(c)
                counts[neighbors[c]] -= 1
                code.append(best_w)
                counts[neighbors[best_w]] += 1
                improved = True

    # Final cleanup
    code = list(set(code))
    counts = np.zeros(N, dtype=np.int16)
    for idx in code:
        counts[neighbors[idx]] += 1
        
    changed = True
    while changed:
        changed = False
        for c in list(code):
            if np.min(counts[neighbors[c]]) > 1:
                code.remove(c)
                counts[neighbors[c]] -= 1
                changed = True

    return [list(words[i]) for i in code]
