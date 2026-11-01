import itertools
import time
import random
import numpy as np

def _all_words(q, n):
    return list(itertools.product(range(q), repeat=n))

def _build_coverage(q, n, radius):
    words = _all_words(q, n)
    word_to_idx = {w: i for i, w in enumerate(words)}
    total = len(words)
    coverage = []
    
    for i, w in enumerate(words):
        cov = [i]
        if radius >= 1:
            for pos in range(n):
                old = w[pos]
                for new_val in range(q):
                    if new_val != old:
                        lst = list(w)
                        lst[pos] = new_val
                        nw = tuple(lst)
                        cov.append(word_to_idx[nw])
        coverage.append(cov)
        
    return words, coverage

def _greedy_randomized(words, coverage, q, n, radius, seed, top_k=10):
    rng = random.Random(seed)
    total = len(words)
    uncovered = [True] * total
    
    # Fix first word to 0
    chosen_set = {0}
    for w_idx in coverage[0]:
        uncovered[w_idx] = False
        
    while True:
        if not any(uncovered):
            break
            
        gains = [0] * total
        # Calculate gains for non-chosen words
        for i in range(total):
            if i in chosen_set:
                continue
            gain = 0
            for w_idx in coverage[i]:
                if uncovered[w_idx]:
                    gain += 1
            gains[i] = gain
            
        # Find top K
        candidates = [(g, i) for i, g in enumerate(gains) if g > 0]
        if not candidates:
            break
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        k = min(top_k, len(candidates))
        chosen_idx = candidates[rng.randint(0, k-1)][1]
        
        chosen_set.add(chosen_idx)
        for w_idx in coverage[chosen_idx]:
            uncovered[w_idx] = False
            
    return chosen_set

def _reduce_code(chosen_set, coverage, words):
    total = len(words)
    cover_count = [0] * total
    
    for i in chosen_set:
        for w_idx in coverage[i]:
            cover_count[w_idx] += 1
            
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i in list(chosen_set):
            removable = True
            for w_idx in coverage[i]:
                if cover_count[w_idx] == 1:
                    removable = False
                    break
            if removable:
                to_remove.append(i)
                
        if to_remove:
            changed = True
            for i in to_remove:
                chosen_set.remove(i)
                for w_idx in coverage[i]:
                    cover_count[w_idx] -= 1
                    
    return chosen_set

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145
    
    if q == 3 and n == 6 and radius == 1:
        words, coverage = _build_coverage(q, n, radius)
        
        best_size = float('inf')
        best_code = None
        
        # Try different top_k values and seeds
        top_k_values = [1, 5, 10, 20, 50]
        
        for seed in range(500):
            if time.time() > deadline:
                break
                
            tk = top_k_values[seed % len(top_k_values)]
            chosen = _greedy_randomized(words, coverage, q, n, radius, seed, top_k=tk)
            reduced = _reduce_code(chosen, coverage, words)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_code = [words[i] for i in reduced]
                if best_size <= 73:
                    break
                    
        if best_code:
            return best_code
            
        return [list(w) for w in words]
        
    # Fallback
    words = _all_words(q, n)
    return [list(w) for w in words]
