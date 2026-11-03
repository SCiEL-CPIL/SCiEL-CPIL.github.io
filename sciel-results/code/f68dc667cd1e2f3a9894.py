"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search with multiple restarts.
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
    
    def greedy_init():
        covered = [False] * total_words
        codewords = []
        uncovered_count = total_words
        
        while uncovered_count > 0:
            # Sample a subset of candidates to speed up greedy selection
            if uncovered_count > total_words * 0.1:
                # Sample a fraction of the space
                sample_size = min(1000, total_words // 10)
                candidates = [random.randint(0, total_words - 1) for _ in range(sample_size)]
            else:
                candidates = list(range(total_words))
            
            best_word = -1
            best_new_coverage = -1
            
            for w in candidates:
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
            
            if best_word == -1:
                # Fallback: pick any uncovered word
                for w in range(total_words):
                    if not covered[w]:
                        best_word = w
                        break
                if best_word == -1:
                    break
            
            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1
        
        return codewords
    
    def optimize(codewords, time_limit):
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
                
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
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        no_improve_count = 0
        best_len = len(codewords)
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                no_improve_count = 0
                if len(codewords) < best_len:
                    best_len = len(codewords)
                continue
                
            if len(codewords) == 0:
                break
            
            # Try multiple swap attempts per iteration
            found_swap = False
            for _ in range(3):
                idx = random.randint(0, len(codewords) - 1)
                old_cw = codewords[idx]
                
                would_uncover = []
                for u in neighbors[old_cw]:
                    if coverage_count[u] == 1:
                        would_uncover.append(u)
                
                if len(would_uncover) == 0:
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    codewords.pop(idx)
                    cw_set.discard(old_cw)
                    found_swap = True
                    break
                
                # Find best candidate from neighbors of uncovered words
                if would_uncover:
                    # Pick a random uncovered word from would_uncover
                    target = random.choice(would_uncover)
                    potential_candidates = list(neighbors[target])
                    random.shuffle(potential_candidates)
                    
                    best_new_cw = -1
                    best_score = -1
                    
                    for new_cw in potential_candidates:
                        if new_cw in cw_set or new_cw == old_cw:
                            continue
                        
                        # Check if new_cw covers all would_uncover
                        valid = True
                        for u in would_uncover:
                            if u not in neighbors[new_cw]:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Score: Maximize coverage of already-covered words (redundancy)
                        score = 0
                        for u in neighbors[new_cw]:
                            if u not in neighbors[old_cw]:
                                if coverage_count[u] >= 1:
                                    score += 1
                        
                        if score > best_score:
                            best_score = score
                            best_new_cw = new_cw
                    
                    if best_new_cw != -1:
                        for u in neighbors[old_cw]:
                            coverage_count[u] -= 1
                        for u in neighbors[best_new_cw]:
                            coverage_count[u] += 1
                        codewords[idx] = best_new_cw
                        cw_set.discard(old_cw)
                        cw_set.add(best_new_cw)
                        found_swap = True
                        break
            
            if not found_swap:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            else:
                no_improve_count = 0
                if len(codewords) < best_len:
                    best_len = len(codewords)
        
        try_remove_redundant()
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    
    while time.time() - start_time < time_limit:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            break
        
        # Allocate time: give more time to later restarts if we're making progress
        iter_time = min(15, remaining_time - 3)
        if iter_time <= 0:
            break
        
        cw = greedy_init()
        cw = optimize(cw, iter_time - 2) # Leave 2s for safety
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
