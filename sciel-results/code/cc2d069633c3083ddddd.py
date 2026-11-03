"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search with multiple restarts, vectorized NumPy operations,
    and aggressive 1-swap moves to escape local optima.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    # Precompute neighbors using vectorized approach
    all_words = np.arange(total_words)
    
    # Convert all words to array form
    word_array = np.zeros((total_words, n), dtype=np.int32)
    for i in range(total_words):
        val = i
        for pos in range(n):
            word_array[i, pos] = val % q
            val //= q
    
    # Precompute neighbors for radius 1 (most common case)
    if radius == 1:
        # For each word, neighbors are itself + all words differing in exactly 1 position
        neighbor_list = [None] * total_words
        for w in range(total_words):
            neigh = [w]
            for pos in range(n):
                orig_val = word_array[w, pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = word_array[w].copy()
                    new_word[pos] = new_val
                    new_int = 0
                    for i in range(n-1, -1, -1):
                        new_int = new_int * q + new_word[i]
                    neigh.append(new_int)
            neighbor_list[w] = np.array(neigh, dtype=np.int32)
    else:
        # Fallback for other radii
        neighbor_list = [None] * total_words
        for w in range(total_words):
            word = int_to_word(w, n, q)
            neigh = [w]
            if radius >= 1:
                for pos in range(n):
                    for val in range(q):
                        if val == word[pos]:
                            continue
                        new_word = word[:]
                        new_word[pos] = val
                        neigh.append(word_to_int(new_word, q))
            if radius >= 2:
                for pos1 in range(n):
                    for pos2 in range(pos1 + 1, n):
                        for val1 in range(q):
                            if val1 == word[pos1]:
                                continue
                            for val2 in range(q):
                                if val2 == word[pos2]:
                                    continue
                                new_word = word[:]
                                new_word[pos1] = val1
                                new_word[pos2] = val2
                                new_int = word_to_int(new_word, q)
                                if new_int not in neigh:
                                    neigh.append(new_int)
            neighbor_list[w] = np.array(neigh, dtype=np.int32)
    
    def greedy_init():
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered_count = total_words
        
        # Use a priority queue approach: repeatedly pick the word that covers most uncovered
        # For speed, we'll use a randomized greedy
        remaining = np.arange(total_words)
        random.shuffle(remaining)
        
        while uncovered_count > 0:
            best_word = -1
            best_new_coverage = -1
            
            # Sample a subset of remaining words for speed
            sample_size = min(len(remaining), 100)
            sample_idx = np.random.choice(len(remaining), size=sample_size, replace=False)
            sample = remaining[sample_idx]
            
            for w in sample:
                neigh = neighbor_list[w]
                new_cov = np.sum(~covered[neigh])
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
                if new_cov == uncovered_count:
                    break
            
            if best_word == -1:
                # Fallback: pick from full remaining
                for w in remaining:
                    neigh = neighbor_list[w]
                    new_cov = np.sum(~covered[neigh])
                    if new_cov > best_new_coverage:
                        best_new_coverage = new_cov
                        best_word = w
                    if new_cov == uncovered_count:
                        break
            
            if best_word == -1:
                break
            
            codewords.append(best_word)
            neigh = neighbor_list[best_word]
            newly_covered = ~covered[neigh]
            covered[neigh] = True
            uncovered_count -= np.sum(newly_covered)
            
            # Remove covered words from remaining
            remaining = remaining[~np.isin(remaining, neigh)]
        
        return codewords
    
    def optimize(codewords, time_limit):
        if not codewords:
            return codewords
        
        # Coverage count array
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage_count[neighbor_list[cw]] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    neigh = neighbor_list[cw]
                    if np.all(coverage_count[neigh] > 1):
                        coverage_count[neigh] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
                
            if len(codewords) == 0:
                break
            
            # Try a random swap
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            old_neigh = neighbor_list[old_cw]
            
            # Find words that would become uncovered if we remove old_cw
            would_uncover = np.where(coverage_count[old_neigh] == 1)[0]
            
            if len(would_uncover) == 0:
                # Can remove without replacement
                coverage_count[old_neigh] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Find candidate replacement: must cover all would_uncover
            # Sample candidates from neighbors of would_uncover words
            candidate_pool = set()
            for u in would_uncover[:5]:  # Limit to first 5 for speed
                candidate_pool.update(neighbor_list[u].tolist())
            
            candidates = [c for c in candidate_pool if c not in cw_set and c != old_cw]
            random.shuffle(candidates)
            
            best_new_cw = -1
            best_score = -1
            
            # Evaluate top candidates
            for new_cw in candidates[:50]:
                new_neigh = neighbor_list[new_cw]
                
                # Check if new_cw covers all would_uncover
                if not np.all(np.isin(would_uncover, new_neigh)):
                    continue
                
                # Score: net change in unique coverage
                # Remove old: coverage_count[old_neigh] -= 1
                # Add new: coverage_count[new_neigh] += 1
                # Count how many words go from count 1 to count 0 (bad) vs count 0 to count 1 (good)
                
                # Words in old_neigh but not in new_neigh: lose coverage
                only_in_old = np.setdiff1d(old_neigh, new_neigh)
                # Words in new_neigh but not in old_neigh: gain coverage  
                only_in_new = np.setdiff1d(new_neigh, old_neigh)
                
                # Words in both: no change
                # Words in neither: no change
                
                # Score: minimize words that become uncovered, maximize words that become covered
                lose = np.sum(coverage_count[only_in_old] == 1)
                gain = np.sum(coverage_count[only_in_new] == 0)
                
                score = gain - lose
                
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1 and best_score >= 0:
                coverage_count[old_neigh] -= 1
                coverage_count[neighbor_list[best_new_cw]] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
        
        try_remove_redundant()
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    best_len = float('inf')
    
    while time.time() - start_time < time_limit:
        # Allocate time for this iteration
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 5:
            break
        
        iter_time = min(30, remaining_time - 5)
        
        cw = greedy_init()
        cw = optimize(cw, iter_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
        
        # If we've found a good solution, try to improve it further
        if best_len < 500:
            # Give more time to optimization of the best solution
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time > 10:
                best_codewords = optimize(best_codewords, remaining_time - 5)
                if len(best_codewords) < best_len:
                    best_len = len(best_codewords)
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
