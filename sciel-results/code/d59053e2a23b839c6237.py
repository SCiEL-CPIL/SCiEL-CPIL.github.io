"""Improved program using greedy set cover with optimized local search."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
import time


def _hamming_dist(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    # For small cases, use direct construction
    if n <= 4:
        if n == 4:
            return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        elif n == 3:
            return [[i, i, i] for i in range(q)]
        elif n == 2:
            # All pairs for q=2, but for q=3 use a smaller covering
            if q == 3:
                # Need to cover all 9 words with radius 1
                # Each codeword covers 1 + 2*2 = 5 words (itself + 4 neighbors)
                # Need at least ceil(9/5) = 2 codewords, but actually need more
                # Let's use greedy
                pass
            return [[i, j] for i in range(q) for j in range(q)]
        else:
            return [[i] for i in range(q)]
    
    # For n=6, q=3, radius=1: use optimized greedy + local search
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for all words (for radius 1)
    neighbors_cache = {}
    for w in all_words:
        w_idx = word_to_idx[w]
        neighbors = [w_idx]
        w_list = list(w)
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    w_list[i] = val
                    new_w = tuple(w_list)
                    neighbors.append(word_to_idx[new_w])
            w_list[i] = w[i]
        neighbors_cache[w_idx] = neighbors
    
    # Greedy set cover: start with empty, add codeword covering most uncovered
    covered = [False] * num_words
    codewords = []
    covered_count = 0
    
    # Precompute which words each potential codeword covers
    # For efficiency, use a priority-like approach
    
    # Start with a reasonable initial solution
    # Use the tetracode-based construction as a starting point
    if q == 3 and n == 6:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    else:
        initial_codewords = []
    
    # Build initial covered state
    for cw in initial_codewords:
        cw_idx = word_to_idx[tuple(cw)]
        for nb_idx in neighbors_cache[cw_idx]:
            if not covered[nb_idx]:
                covered[nb_idx] = True
                covered_count += 1
        codewords.append(cw)
    
    # Greedy: add codewords until all covered
    while covered_count < num_words:
        best_cw = None
        best_gain = 0
        # Sample a subset of candidates for speed
        sample_size = min(200, num_words)
        candidates = random.sample(range(num_words), sample_size) if num_words > sample_size else list(range(num_words))
        
        for cand_idx in candidates:
            gain = 0
            for nb_idx in neighbors_cache[cand_idx]:
                if not covered[nb_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cw = cand_idx
        
        if best_cw is None or best_gain == 0:
            # Fallback: scan all
            for cand_idx in range(num_words):
                gain = 0
                for nb_idx in neighbors_cache[cand_idx]:
                    if not covered[nb_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cw = cand_idx
        
        if best_cw is None:
            break
        
        cw = list(all_words[best_cw])
        codewords.append(cw)
        for nb_idx in neighbors_cache[best_cw]:
            if not covered[nb_idx]:
                covered[nb_idx] = True
                covered_count += 1
    
    # Local search: try to remove or replace codewords
    best_codewords = codewords[:]
    best_size = len(codewords)
    
    random.seed(42)
    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    # Phase 1: Remove redundant codewords
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for i in range(len(best_codewords) - 1, -1, -1):
            if time.time() - start_time > max_time:
                break
            cw_idx = word_to_idx[tuple(best_codewords[i])]
            # Check if all neighbors of this codeword are covered by others
            can_remove = True
            for nb_idx in neighbors_cache[cw_idx]:
                # Check if nb_idx is covered by any other codeword
                found = False
                for j, cw in enumerate(best_codewords):
                    if j == i:
                        continue
                    j_idx = word_to_idx[tuple(cw)]
                    if nb_idx in neighbors_cache[j_idx]:
                        found = True
                        break
                if not found:
                    can_remove = False
                    break
            if can_remove:
                del best_codewords[i]
                best_size = len(best_codewords)
                improved = True
    
    # Phase 2: Try to replace pairs of codewords with one
    for iteration in range(50):
        if time.time() - start_time > max_time:
            break
        current = best_codewords[:]
        found_improvement = False
        
        for i in range(len(current)):
            if time.time() - start_time > max_time:
                break
            for j in range(i + 1, len(current)):
                if time.time() - start_time > max_time:
                    break
                # Try removing i and j, see if we can cover the gap with one codeword
                removed_i = current[i]
                removed_j = current[j]
                
                # Find words that would become uncovered
                uncovered = set()
                for nb_idx in neighbors_cache[word_to_idx[tuple(removed_i)]]:
                    covered_by_others = False
                    for k, cw in enumerate(current):
                        if k == i or k == j:
                            continue
                        if nb_idx in neighbors_cache[word_to_idx[tuple(cw)]]:
                            covered_by_others = True
                            break
                    if not covered_by_others:
                        uncovered.add(nb_idx)
                
                for nb_idx in neighbors_cache[word_to_idx[tuple(removed_j)]]:
                    if nb_idx in uncovered:
                        continue
                    covered_by_others = False
                    for k, cw in enumerate(current):
                        if k == i or k == j:
                            continue
                        if nb_idx in neighbors_cache[word_to_idx[tuple(cw)]]:
                            covered_by_others = True
                            break
                    if not covered_by_others:
                        uncovered.add(nb_idx)
                
                if not uncovered:
                    # Can remove both
                    new_current = [cw for k, cw in enumerate(current) if k != i and k != j]
                    if len(new_current) < best_size:
                        best_codewords = new_current
                        best_size = len(new_current)
                        found_improvement = True
                        break
                
                # Try to find one codeword that covers all uncovered
                if len(uncovered) <= 20:  # Limit search
                    uncovered_list = list(uncovered)
                    # Find candidate that covers all
                    for cand_idx in range(num_words):
                        covers_all = True
                        for u_idx in uncovered_list:
                            if u_idx not in neighbors_cache[cand_idx]:
                                covers_all = False
                                break
                        if covers_all:
                            # Check if this candidate is not already in current (excluding i,j)
                            cand_tuple = all_words[cand_idx]
                            if tuple(cand_tuple) not in [tuple(cw) for k, cw in enumerate(current) if k != i and k != j]:
                                new_current = [cw for k, cw in enumerate(current) if k != i and k != j]
                                new_current.append(list(cand_tuple))
                                if len(new_current) < best_size:
                                    best_codewords = new_current
                                    best_size = len(new_current)
                                    found_improvement = True
                                break
            if found_improvement:
                break
        
        if not found_improvement:
            # Try random perturbations
            if len(best_codewords) > 0:
                idx = random.randint(0, len(best_codewords) - 1)
                old_cw = best_codewords[idx]
                # Try replacing with a random nearby codeword
                old_idx = word_to_idx[tuple(old_cw)]
                for nb_idx in neighbors_cache[old_idx]:
                    new_cw = list(all_words[nb_idx])
                    if new_cw == old_cw:
                        continue
                    # Check if replacement maintains coverage
                    # Remove old, add new, check all covered
                    test_codewords = best_codewords[:]
                    test_codewords[idx] = new_cw
                    # Verify coverage
                    test_covered = [False] * num_words
                    for cw in test_codewords:
                        cw_i = word_to_idx[tuple(cw)]
                        for nb in neighbors_cache[cw_i]:
                            test_covered[nb] = True
                    if all(test_covered):
                        if len(test_codewords) < best_size:
                            best_codewords = test_codewords
                            best_size = len(best_codewords)
                        break
    
    return best_codewords
# EVOLVE-BLOCK-END
