"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search starting from a greedy feasible covering.
    Improved local search with targeted swap candidates.
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
    
    all_words = list(range(total_words))
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
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
        neighbors[w] = set(neigh)
    
    # Greedy initialization
    covered = [False] * total_words
    codewords = []
    uncovered_count = total_words
    remaining = set(all_words)
    
    while uncovered_count > 0:
        best_word = -1
        best_new_coverage = -1
        # Randomize the order of checking to break ties and avoid deterministic local minima
        # We can shuffle remaining if it's not too large, or just iterate and pick best
        # For 6561 words, iterating is fine.
        for w in remaining:
            new_cov = 0
            for u in neighbors[w]:
                if not covered[u]:
                    new_cov += 1
            if new_cov > best_new_coverage:
                best_new_coverage = new_cov
                best_word = w
            if new_cov == uncovered_count:
                break
        if best_word == -1:
            break
        codewords.append(best_word)
        remaining.discard(best_word)
        for u in neighbors[best_word]:
            if not covered[u]:
                covered[u] = True
                uncovered_count -= 1
    
    coverage_count = [0] * total_words
    for cw in codewords:
        for u in neighbors[cw]:
            coverage_count[u] += 1
    
    # Helper to try removing redundant codewords
    def try_remove_redundant():
        removed_any = False
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords) - 1, -1, -1):
                cw = codewords[i]
                can_remove = True
                for u in neighbors[cw]:
                    if coverage_count[u] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for u in neighbors[cw]:
                        coverage_count[u] -= 1
                    codewords.pop(i)
                    changed = True
                    removed_any = True
                    break
        return removed_any
    
    try_remove_redundant()
    
    start_time = time.time()
    time_limit = 115
    
    cw_set = set(codewords)
    
    while time.time() - start_time < time_limit:
        if try_remove_redundant():
            continue
        
        if len(codewords) == 0:
            break
            
        # Try a swap
        idx = random.randint(0, len(codewords) - 1)
        old_cw = codewords[idx]
        
        # Find words uniquely covered by old_cw
        would_uncover = []
        for u in neighbors[old_cw]:
            if coverage_count[u] == 1:
                would_uncover.append(u)
        
        if len(would_uncover) == 0:
            # Should have been caught by try_remove_redundant, but safety check
            for u in neighbors[old_cw]:
                coverage_count[u] -= 1
            codewords.pop(idx)
            cw_set.discard(old_cw)
            continue
        
        # Generate candidate new_cw's
        # 1. Neighbors of old_cw (excluding old_cw itself)
        # 2. If that fails, try neighbors of the words in would_uncover?
        # Let's stick to neighbors of old_cw first as they are the most likely to cover the "gap"
        candidates = list(neighbors[old_cw])
        random.shuffle(candidates)
        
        # Also add a few random global candidates to escape local optima
        for _ in range(5):
            candidates.append(random.randint(0, total_words - 1))
        
        swap_done = False
        for new_cw in candidates:
            if new_cw == old_cw or new_cw in cw_set:
                continue
            
            # Check if new_cw covers all would_uncover words
            valid = True
            for u in would_uncover:
                if u not in neighbors[new_cw]:
                    valid = False
                    break
            
            if valid:
                # Perform swap
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbors[new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = new_cw
                cw_set.discard(old_cw)
                cw_set.add(new_cw)
                swap_done = True
                break
        
        if not swap_done:
            # If no swap found for this old_cw, we can just continue to next iteration
            # This might lead to stagnation. 
            # To improve, we could try a different old_cw or perturb the solution.
            # For now, just loop.
            pass

    try_remove_redundant()
    
    result = [int_to_word(cw, n, q) for cw in codewords]
    return result
# EVOLVE-BLOCK-END
