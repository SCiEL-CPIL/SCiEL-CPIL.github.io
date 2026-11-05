import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        # Greedy Set Cover
        all_words = list(itertools.product(range(q), repeat=n))
        code = []
        covered = set()
        
        for w in all_words:
            if w not in covered:
                code.append(list(w))
                for i in range(n):
                    for alt in range(q):
                        if alt != w[i]:
                            new_w = list(w)
                            new_w[i] = alt
                            covered.add(tuple(new_w))
                covered.add(tuple(w))
        
        return code

    total = 3 ** n
    powers = [3 ** p for p in range(n)]
    
    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    def to_word(idx):
        return [(idx // powers[p]) % 3 for p in range(n)]

    # Precompute neighbors
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(n)]
        neighbors[i, 0] = i
        k = 1
        for p in range(n):
            base = i - d[p] * powers[p]
            for alt in range(3):
                if alt == d[p]:
                    continue
                neighbors[i, k] = base + alt * powers[p]
                k += 1

    # Initial code: Tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_idx = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    best_code = set(init_idx)
    best_size = len(best_code)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def get_coverage(code_set):
        cov = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cov[neighbors[c]] += 1
        return cov

    def greedy_remove(code_set, cov):
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                if time.time() > deadline:
                    return code_set, cov
                if np.all(cov[neighbors[c]] >= 2):
                    code_set.discard(c)
                    cov[neighbors[c]] -= 1
                    changed = True
        return code_set, cov

    def repair(code_set, cov):
        uncovered = np.where(cov == 0)[0]
        while len(uncovered) > 0 and time.time() < deadline:
            if len(uncovered) == 0:
                break
            
            best_gain = -1
            best_cand = -1
            
            # Optimization: Only check candidates that cover at least one uncovered word?
            # We can precompute which candidates cover which uncovered words?
            # Or just check all 729.
            
            for c in range(total):
                if c in code_set:
                    continue
                gain = np.sum(cov[neighbors[c]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_cand = c
            
            if best_cand != -1 and best_gain > 0:
                code_set.add(best_cand)
                cov[neighbors[best_cand]] += 1
                uncovered = np.where(cov == 0)[0]
            else:
                # Should not happen if we have a valid cover, but break to avoid infinite loop
                break
        
        return code_set, cov

    current_code = set(best_code)
    current_cov = get_coverage(current_code)
    
    current_code, current_cov = greedy_remove(current_code, current_cov)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = set(current_code)

    while time.time() < deadline:
        if best_size == 71:
            break

        # Perturbation
        # Increase perturbation size if we are stuck?
        # For now, fixed 1-3
        num_remove = random.choice([1, 2, 3])
        
        if len(current_code) <= num_remove:
            current_code = set(best_code)
            current_cov = get_coverage(current_code)
            continue
            
        to_remove = random.sample(list(current_code), num_remove)
        
        saved_code = set(current_code)
        saved_cov = current_cov.copy()
        
        for c in to_remove:
            current_code.discard(c)
            current_cov[neighbors[c]] -= 1
            
        current_code, current_cov = repair(current_code, current_cov)
        current_code, current_cov = greedy_remove(current_code, current_cov)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = set(current_code)
            
        # If the current code is significantly larger than the best, revert to best
        # This helps avoid wandering in bad regions
        if len(current_code) > best_size + 5:
            current_code = set(best_code)
            current_cov = get_coverage(current_code)
            current_code, current_cov = greedy_remove(current_code, current_cov)

    return [to_word(c) for c in best_code]
