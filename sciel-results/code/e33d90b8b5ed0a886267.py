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
    
    # Precompute neighbors as lists of ints for faster iteration
    neighbors = [None] * total_words
    for w in all_words:
        word = int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.append(word_to_int(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1 = word[pos1]
                    orig2 = word[pos2]
                    for val1 in range(q):
                        if val1 == orig1:
                            continue
                        for val2 in range(q):
                            if val2 == orig2:
                                continue
                            new_word = word[:]
                            new_word[pos1] = val1
                            new_word[pos2] = val2
                            new_int = word_to_int(new_word, q)
                            if new_int not in neigh:
                                neigh.append(new_int)
        neighbors[w] = neigh
        
    # Precompute reverse neighbors (who covers w) for faster greedy? 
    # Actually, for greedy we pick an uncovered word u, and need candidates c such that u in neighbors[c].
    # This is the reverse neighbor list.
    rev_neighbors = [set() for _ in range(total_words)]
    for w in all_words:
        for u in neighbors[w]:
            rev_neighbors[u].add(w)

    def greedy_init():
        covered = [False] * total_words
        codewords = []
        uncovered_count = total_words
        # Use a list of uncovered words for random selection
        uncovered_list = list(all_words)
        
        while uncovered_count > 0:
            # Pick a random uncovered word
            idx = random.randint(0, len(uncovered_list) - 1)
            u = uncovered_list[idx]
            
            # Get candidates that cover u
            candidates = list(rev_neighbors[u])
            random.shuffle(candidates)
            
            best_word = -1
            best_new_coverage = -1
            
            for w in candidates:
                new_cov = 0
                for u2 in neighbors[w]:
                    if not covered[u2]:
                        new_cov += 1
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
                    if new_cov == len(neighbors[w]): # Max possible for this word
                        break
            
            if best_word == -1:
                # Should not happen if coverage is valid
                break
            
            codewords.append(best_word)
            
            # Update covered status
            for u2 in neighbors[best_word]:
                if not covered[u2]:
                    covered[u2] = True
                    uncovered_count -= 1
            
            # Remove covered words from uncovered_list? 
            # Instead of maintaining a dynamic list which is O(N) to filter, 
            # we can just pick random indices and check if covered.
            # But we need to ensure we pick an *uncovered* one.
            # Simple approach: keep a set of uncovered indices or just re-scan.
            # Given N=6561, re-scanning is fast.
            # Let's just pick a random word from all_words until we find an uncovered one.
            # This is expected O(1) if coverage is high, O(N) if low.
            pass
        
        return codewords

    # Improved Greedy: maintain a set of uncovered words
    def greedy_init_fast():
        covered = [False] * total_words
        codewords = []
        uncovered_set = set(all_words)
        
        while uncovered_set:
            # Pick random uncovered word
            u = random.choice(list(uncovered_set))
            
            # Candidates covering u
            candidates = list(rev_neighbors[u])
            random.shuffle(candidates)
            
            best_word = -1
            best_new_coverage = -1
            
            for w in candidates:
                new_cov = 0
                for u2 in neighbors[w]:
                    if u2 in uncovered_set:
                        new_cov += 1
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
            
            if best_word == -1:
                break
                
            codewords.append(best_word)
            
            # Update uncovered_set
            for u2 in neighbors[best_word]:
                if u2 in uncovered_set:
                    uncovered_set.remove(u2)
        
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
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
                
            if len(codewords) == 0:
                break
                
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find private words of old_cw
            private_words = []
            for u in neighbors[old_cw]:
                if coverage_count[u] == 1:
                    private_words.append(u)
            
            if len(private_words) == 0:
                # Can remove directly
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Find a replacement that covers all private_words
            # Intersect neighbors of all private words
            # Start with neighbors of the first private word
            potential_candidates = set(neighbors[private_words[0]])
            
            for pw in private_words[1:]:
                potential_candidates.intersection_update(neighbors[pw])
                if not potential_candidates:
                    break
            
            if not potential_candidates:
                continue
            
            # Filter out current codewords and old_cw
            valid_candidates = [c for c in potential_candidates if c not in cw_set and c != old_cw]
            
            if not valid_candidates:
                continue
            
            random.shuffle(valid_candidates)
            
            best_new_cw = -1
            best_score = -1
            
            for new_cw in valid_candidates:
                # Score: how many new words does it cover that were not covered by old_cw?
                # Or better: how many words does it cover that are covered by multiple others?
                # Actually, we just need it to cover private_words.
                # To maximize future removals, we want new_cw to cover words that are already well-covered?
                # No, we want to create a situation where some other codeword becomes redundant.
                # A good heuristic: minimize the number of private words of new_cw?
                # Or maximize the overlap with other codewords?
                
                # Let's score by: sum of (coverage_count[u] - 1) for u in neighbors[new_cw]
                # This rewards covering words that are already covered by others (redundancy).
                score = 0
                for u in neighbors[new_cw]:
                    if u != old_cw: # old_cw is being removed
                        score += coverage_count[u] - 1
                
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1:
                # Perform swap
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbors[best_new_cw]:
                    coverage_count[u] += 1
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
        iter_time = min(10, time_limit - (time.time() - start_time))
        if iter_time <= 1:
            break
        
        cw = greedy_init_fast()
        # Optimize with remaining time for this restart
        opt_time = iter_time - 1
        if opt_time > 0:
            cw = optimize(cw, opt_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
            
        # Early stop if we hit a known good bound? 
        # Best known is 486. If we reach 486, we can stop? 
        # Or keep going to find 485?
        # Let's keep going but maybe reduce restarts if we are stable.
        
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
