import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    if q != 3 or n != 8 or radius != 1:
        return _general_greedy(q, n, radius, deadline - start_time)

    total_words = 3 ** n
    # Precompute balls: for each word, the set of words within distance 1
    balls = np.empty((total_words, 17), dtype=np.int32)
    
    def int_to_base3(x, n):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = x % 3
            x //= 3
        return digits

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    for x in range(total_words):
        digits = int_to_base3(x, n)
        ball_indices = [x]
        for j in range(n):
            for v in range(3):
                if v != digits[j]:
                    new_digits = digits[:]
                    new_digits[j] = v
                    ball_indices.append(base3_to_int(new_digits))
        balls[x] = ball_indices

    # Also create neighbor sets for fast membership testing in local search
    neighbor_sets = [set(balls[i]) for i in range(total_words)]

    remaining = np.ones(total_words, dtype=bool)
    codewords = []
    
    # Greedy Set Cover
    while remaining.any():
        if time.time() > deadline - 30:
            break
            
        # Vectorized count of uncovered vertices (words) in each hyperedge (ball)
        counts = remaining[balls].sum(axis=1)
        best_candidate = np.argmax(counts)
        best_count = counts[best_candidate]
        
        if best_count == 0:
            break
            
        codewords.append(best_candidate)
        remaining[balls[best_candidate]] = False

    # Redundancy Removal
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in codewords:
        cover_count[balls[c]] += 1
        
    changed = True
    while changed:
        if time.time() > deadline - 25:
            break
        changed = False
        new_codewords = []
        for c in codewords:
            if cover_count[balls[c]].min() >= 2:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_codewords.append(c)
        codewords = new_codewords

    # Local Search: 1-swap and 2-swap operations
    code_set = set(codewords)
    
    def compute_cov_count(code_set):
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            cov_count[balls[c]] += 1
        return cov_count
    
    cov_count = compute_cov_count(code_set)
    
    improved = True
    max_local_iters = 50
    local_iter = 0
    
    while improved and time.time() < deadline - 10 and local_iter < max_local_iters:
        improved = False
        local_iter += 1
        c_list = list(code_set)
        random.shuffle(c_list)
        
        # Try 1-swaps: remove c, add d if all words in N(c) are still covered
        for c in c_list:
            if time.time() > deadline - 10:
                break
            if c not in code_set:
                continue
            
            # Find words that would become uncovered if c is removed
            # These are words in N(c) with cov_count[w] == 1
            critical_words = [w for w in balls[c] if cov_count[w] == 1]
            
            if not critical_words:
                # c is redundant, remove it
                code_set.remove(c)
                cov_count[balls[c]] -= 1
                improved = True
                continue
            
            # Try to find d that covers all critical_words
            # d must be within distance 1 of all critical_words
            # This means d is in the intersection of N(w) for all w in critical_words
            
            # For efficiency, start with N(critical_words[0]) and intersect
            if len(critical_words) > 3:
                # Too many critical words, unlikely to find a single d
                continue
                
            candidate_set = set(balls[critical_words[0]])
            for w in critical_words[1:]:
                candidate_set &= set(balls[w])
            
            # Find a candidate d that is not already in code_set
            # and covers all critical_words
            for d in candidate_set:
                if d in code_set:
                    continue
                
                # Check if d covers all critical_words
                valid = True
                for w in critical_words:
                    if w not in neighbor_sets[d]:
                        valid = False
                        break
                
                if valid:
                    # Perform the swap
                    code_set.remove(c)
                    code_set.add(d)
                    cov_count[balls[c]] -= 1
                    cov_count[balls[d]] += 1
                    improved = True
                    break
        
        # Try 2-swaps: remove c1, c2, add d1, d2 if all words in N(c1)∪N(c2) are covered
        if not improved and time.time() < deadline - 15:
            c_list = list(code_set)
            random.shuffle(c_list)
            
            # Limit 2-swap attempts
            for i in range(min(len(c_list), 20)):
                if time.time() > deadline - 15:
                    break
                c1 = c_list[i]
                if c1 not in code_set:
                    continue
                
                # Find critical words for c1
                critical1 = [w for w in balls[c1] if cov_count[w] == 1]
                if len(critical1) > 2:
                    continue
                
                for j in range(i+1, min(len(c_list), i+20)):
                    if time.time() > deadline - 15:
                        break
                    c2 = c_list[j]
                    if c2 not in code_set:
                        continue
                    
                    # Find critical words for c2
                    critical2 = [w for w in balls[c2] if cov_count[w] == 1]
                    
                    # Combined critical words: words that would become uncovered
                    # A word w is critical if it's in N(c1)∪N(c2) and cov_count[w] == 1
                    # (after removing both c1 and c2)
                    
                    # Temporarily remove c1 and c2 to see what's uncovered
                    temp_cov = cov_count.copy()
                    temp_cov[balls[c1]] -= 1
                    temp_cov[balls[c2]] -= 1
                    
                    critical_all = [w for w in set(balls[c1]) | set(balls[c2]) if temp_cov[w] <= 0]
                    
                    if len(critical_all) > 4:
                        continue
                    
                    # Try to find d1, d2 that cover all critical_all
                    # This is more complex; try simple approach
                    if not critical_all:
                        # Both c1 and c2 are redundant
                        code_set.remove(c1)
                        code_set.remove(c2)
                        cov_count[balls[c1]] -= 1
                        cov_count[balls[c2]] -= 1
                        improved = True
                        break
                    
                    # Try to find a pair (d1, d2) that covers critical_all
                    # Start with candidates from critical_all[0]
                    if critical_all:
                        candidates = set(balls[critical_all[0]])
                        for w in critical_all[1:]:
                            candidates &= set(balls[w])
                        
                        # Try single d that covers all (if possible)
                        for d in candidates:
                            if d in code_set:
                                continue
                            valid = True
                            for w in critical_all:
                                if w not in neighbor_sets[d]:
                                    valid = False
                                    break
                            if valid:
                                code_set.remove(c1)
                                code_set.remove(c2)
                                code_set.add(d)
                                cov_count[balls[c1]] -= 1
                                cov_count[balls[c2]] -= 1
                                cov_count[balls[d]] += 1
                                improved = True
                                break
                        
                        if improved:
                            break
                        
                        # Try pair (d1, d2)
                        # Limit search space
                        candidate_list = list(candidates)[:10]
                        for d1 in candidate_list:
                            if time.time() > deadline - 15:
                                break
                            if d1 in code_set:
                                continue
                            
                            # Words still uncovered after adding d1
                            remaining_critical = [w for w in critical_all if w not in neighbor_sets[d1]]
                            
                            if not remaining_critical:
                                code_set.remove(c1)
                                code_set.remove(c2)
                                code_set.add(d1)
                                cov_count[balls[c1]] -= 1
                                cov_count[balls[c2]] -= 1
                                cov_count[balls[d1]] += 1
                                improved = True
                                break
                            
                            # Find d2 that covers remaining_critical
                            if remaining_critical:
                                candidates2 = set(balls[remaining_critical[0]])
                                for w in remaining_critical[1:]:
                                    candidates2 &= set(balls[w])
                                
                                for d2 in candidates2:
                                    if d2 in code_set or d2 == d1:
                                        continue
                                    valid = True
                                    for w in remaining_critical:
                                        if w not in neighbor_sets[d2]:
                                            valid = False
                                            break
                                    if valid:
                                        code_set.remove(c1)
                                        code_set.remove(c2)
                                        code_set.add(d1)
                                        code_set.add(d2)
                                        cov_count[balls[c1]] -= 1
                                        cov_count[balls[c2]] -= 1
                                        cov_count[balls[d1]] += 1
                                        cov_count[balls[d2]] += 1
                                        improved = True
                                        break
                                
                                if improved:
                                    break
                        
                        if improved:
                            break

    # Final redundancy removal
    cov_count = compute_cov_count(code_set)
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        new_code_set = set()
        for c in code_set:
            if cov_count[balls[c]].min() >= 2:
                cov_count[balls[c]] -= 1
                changed = True
            else:
                new_code_set.add(c)
        code_set = new_code_set

    result = [int_to_base3(c, n) for c in code_set]
    
    # Verify coverage and fix if needed
    remaining = np.ones(total_words, dtype=bool)
    for c in code_set:
        remaining[balls[c]] = False
    
    if remaining.any():
        # Add uncovered words as codewords
        for idx in np.where(remaining)[0]:
            if time.time() > deadline - 1:
                break
            code_set.add(idx)
            remaining[balls[idx]] = False
        result = [int_to_base3(c, n) for c in code_set]

    return result

def _general_greedy(q, n, radius, time_limit):
    # Simple fallback for non-3,8,1 cases
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w

    def word_to_int(w):
        res = 0
        for d in w:
            res = res * q + d
        return res

    start = time.time()
    while covered.sum() < total:
        if time.time() - start > time_limit - 5:
            break
        best_c = -1
        best_count = -1
        for c in range(total):
            if covered[c]: continue
            w = int_to_word(c)
            count = 0
            if not covered[c]: count += 1
            for j in range(n):
                for v in range(q):
                    if v == w[j]: continue
                    w2 = w[:]
                    w2[j] = v
                    idx = word_to_int(w2)
                    if not covered[idx]: count += 1
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c == -1: break
        codewords.append(best_c)
        w = int_to_word(best_c)
        covered[best_c] = True
        for j in range(n):
            for v in range(q):
                if v == w[j]: continue
                w2 = w[:]
                w2[j] = v
                covered[word_to_int(w2)] = True
        
    return [int_to_word(c) for c in codewords]
