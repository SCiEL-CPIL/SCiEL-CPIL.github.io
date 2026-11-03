"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses a fast greedy construction with neighborhood-based candidate sampling,
    followed by randomized local search that swaps codewords to maximize redundant
    coverage, with periodic redundancy removal.
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
    
    # Precompute neighbors efficiently
    # For q=3, n=8, radius=1: each word has 1 + 8*2 = 17 neighbors
    # Total: 6561 * 17 = ~111537 entries, very manageable
    neighbors = [None] * total_words
    neighbors_arr = [None] * total_words  # numpy arrays for vectorized ops
    
    # Precompute word arrays for fast neighbor computation
    if radius == 1:
        # Fast path for radius=1
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_int(word, q))
                word[pos] = cur
            neighbors[w] = set(neigh)
            neighbors_arr[w] = np.array(neigh, dtype=np.int32)
    elif radius == 2:
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_int(word, q))
                    word[pos] = cur
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_int(word, q))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = set(neigh)
            neighbors_arr[w] = np.array(neigh, dtype=np.int32)
    else:
        # General case
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_int(word, q))
                    word[pos] = cur
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_int(word, q))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = set(neigh)
            neighbors_arr[w] = np.array(neigh, dtype=np.int32)
    
    # Also build reverse neighbors: for each word, which codewords cover it
    # This is just the same as neighbors but from the codeword perspective
    # neighbors[cw] = set of words covered by cw
    
    def greedy_init():
        """Fast greedy: repeatedly pick the word covering most uncovered words.
        Uses neighborhood-based candidate sampling for speed."""
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered_count = total_words
        
        # For efficiency, track uncovered words in a set
        uncovered = set(range(total_words))
        
        while uncovered:
            best_word = -1
            best_new_cov = -1
            
            # Sample from uncovered words to find a good candidate
            # The candidate should be in the neighborhood of uncovered words
            # Pick a random uncovered word and look at its neighbors as candidates
            if len(uncovered) > 500:
                # Sample some uncovered words
                sample_uncovered = random.sample(list(uncovered), min(100, len(uncovered)))
            else:
                sample_uncovered = list(uncovered)
            
            # Collect candidate codewords: neighbors of sampled uncovered words
            candidates = set()
            for u in sample_uncovered:
                candidates.update(neighbors[u])
            
            if not candidates:
                break
            
            # Evaluate candidates using numpy for speed
            for w in candidates:
                neigh_arr = neighbors_arr[w]
                new_cov = np.sum(covered[neigh_arr] == False)
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_word = w
                    if new_cov == len(uncovered):
                        break
            
            if best_word == -1:
                # Fallback: pick any uncovered word's neighbor
                u = next(iter(uncovered))
                best_word = next(iter(neighbors[u]))
            
            codewords.append(best_word)
            # Update covered using numpy
            covered[neighbors_arr[best_word]] = True
            # Update uncovered set
            for u in neighbors[best_word]:
                if u in uncovered:
                    uncovered.discard(u)
            
            uncovered_count -= best_new_cov if best_new_cov > 0 else 1
        
        return codewords
    
    def optimize(codewords, time_limit):
        """Local search: remove redundant codewords, swap codewords to improve coverage."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage_count[neighbors_arr[cw]] += 1
        
        cw_set = set(codewords)
        
        def remove_redundant():
            """Remove codewords whose removal doesn't uncover any word."""
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    # Vectorized check: no word in N(cw) has coverage_count == 1
                    if np.all(coverage_count[neighbors_arr[cw]] > 1):
                        coverage_count[neighbors_arr[cw]] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        # Initial redundancy removal
        remove_redundant()
        
        start_time = time.time()
        last_redundancy_check = start_time
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() - start_time < time_limit:
            if no_improve_count >= max_no_improve:
                break
                
            # Periodically try to remove redundant codewords
            if time.time() - last_redundancy_check > 0.2:
                if remove_redundant():
                    no_improve_count = 0
                last_redundancy_check = time.time()
                continue
            
            if not codewords:
                break
            
            # Pick a codeword to try swapping/removing
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]
            old_neigh_arr = neighbors_arr[old_cw]
            
            # Words uniquely covered by old_cw (vectorized)
            would_uncover_mask = coverage_count[old_neigh_arr] == 1
            if not np.any(would_uncover_mask):
                # Can remove this codeword
                coverage_count[old_neigh_arr] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                no_improve_count = 0
                continue
            
            would_uncover = old_neigh_arr[would_uncover_mask]
            
            # Need to find a replacement that covers all would_uncover words
            # Candidate pool: intersection of neighbors of all would_uncover words
            # For efficiency, start with neighbors of the first and filter
            first_uncovered = would_uncover[0]
            candidates = list(neighbors[first_uncovered])
            random.shuffle(candidates)
            
            best_new_cw = -1
            best_score = -1
            
            # Limit candidates to speed up
            max_candidates = 30
            for new_cw in candidates[:max_candidates]:
                if new_cw in cw_set or new_cw == old_cw:
                    continue
                
                new_neigh_arr = neighbors_arr[new_cw]
                new_neigh_set = neighbors[new_cw]
                
                # Check if new_cw covers all would_uncover (vectorized)
                if not np.all(np.isin(would_uncover, new_neigh_arr)):
                    continue
                
                # Check that removing old_cw doesn't create new uncovered words
                # Words in N(old) but not N(new) with coverage_count == 1 would become uncovered
                old_not_in_new = np.setdiff1d(old_neigh_arr, new_neigh_arr, assume_unique=True)
                if np.any(coverage_count[old_not_in_new] == 1):
                    continue
                
                # Score: how many new redundancies we create
                # Words in N(new) but not N(old) that are already covered
                new_not_in_old = np.setdiff1d(new_neigh_arr, old_neigh_arr, assume_unique=True)
                score = np.sum(coverage_count[new_not_in_old] > 0)
                
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1:
                # Perform swap
                coverage_count[old_neigh_arr] -= 1
                coverage_count[neighbors_arr[best_new_cw]] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
                no_improve_count = 0
            else:
                no_improve_count += 1
                # No good swap found. Try to add a random codeword and then remove a redundant one?
                # This increases size temporarily but may escape local minima.
                # Try with some probability to avoid too much overhead
                if random.random() < 0.05 and len(codewords) < 550:
                    # Pick a random word as potential new codeword
                    new_cw = random.randrange(total_words)
                    if new_cw not in cw_set:
                        # Add it temporarily
                        coverage_count[neighbors_arr[new_cw]] += 1
                        codewords.append(new_cw)
                        cw_set.add(new_cw)
                        
                        # Try to remove a redundant codeword
                        removed = remove_redundant()
                        if not removed:
                            # Revert
                            coverage_count[neighbors_arr[new_cw]] -= 1
                            codewords.pop()
                            cw_set.discard(new_cw)
        
        remove_redundant()
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    
    while time.time() - start_time < time_limit:
        # Allocate time for this iteration
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            break
        iter_time = min(12, remaining - 2)
        if iter_time <= 1:
            break
        
        cw = greedy_init()
        cw = optimize(cw, iter_time - 1)
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
        
        # Early exit if we hit the theoretical lower bound or close to it
        if len(best_codewords) <= 410:
            break
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
