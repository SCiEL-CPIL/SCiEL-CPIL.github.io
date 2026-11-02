import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Specifics for the task
    total_words = q ** n
    num_coords = n
    alphabet = q

    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits

    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx

    # Precompute coverage: coverage[cw] = list of words covered by cw
    cover_size = 1 + num_coords * (alphabet - 1)
    coverage = np.zeros((total_words, cover_size), dtype=np.int32)
    for cw_idx in range(total_words):
        digits = idx_to_digits(cw_idx)
        coverage[cw_idx, 0] = cw_idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                coverage[cw_idx, j] = digits_to_idx(new_digits)
                j += 1

    # Precompute reverse coverage: covers_word[w] = list of codewords covering w
    covers_word = np.zeros((total_words, cover_size), dtype=np.int32)
    for w_idx in range(total_words):
        digits = idx_to_digits(w_idx)
        covers_word[w_idx, 0] = w_idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                covers_word[w_idx, j] = digits_to_idx(new_digits)
                j += 1

    # 1. Greedy Initialization with Randomization
    uncovered = np.ones(total_words, dtype=np.uint8)
    count = np.zeros(total_words, dtype=np.int32)
    
    # Initialize counts
    for w_idx in range(total_words):
        for j in range(cover_size):
            cw_idx = covers_word[w_idx, j]
            count[cw_idx] += 1
            
    codewords = []
    num_covered = 0
    
    while num_covered < total_words:
        if time.time() > deadline:
            break
        # Find max count, break ties randomly
        max_count = np.max(count)
        if max_count == 0:
            break
        candidates = np.where(count == max_count)[0]
        best_cw = candidates[random.randint(0, len(candidates) - 1)]
        
        codewords.append(best_cw)
        for j in range(cover_size):
            w_idx = coverage[best_cw, j]
            if uncovered[w_idx]:
                uncovered[w_idx] = 0
                num_covered += 1
                for k in range(cover_size):
                    cw_idx = covers_word[w_idx, k]
                    count[cw_idx] -= 1
                    
    if num_covered < total_words:
        # Fallback if greedy failed (should not happen)
        return [idx_to_digits(cw) for cw in codewords]

    # 2. Local Search / Minimization
    set_of_cw = set(codewords)
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Initialize cover_count
    for cw in set_of_cw:
        for j in range(cover_size):
            w_idx = coverage[cw, j]
            cover_count[w_idx] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Single Removals
        # A codeword is removable if min(cover_count[coverage[cw]]) > 1
        for cw in list(set_of_cw):
            removable = True
            for j in range(cover_size):
                w_idx = coverage[cw, j]
                if cover_count[w_idx] <= 1:
                    removable = False
                    break
            
            if removable:
                set_of_cw.remove(cw)
                for j in range(cover_size):
                    w_idx = coverage[cw, j]
                    cover_count[w_idx] -= 1
                improved = True
                break
        
        if improved:
            continue

        # Phase 2: Pair Swaps (2-1)
        # Try to remove c1, c2 and add c3 such that validity is maintained.
        # Words that become uncovered are those with cover_count == 1 and covered by c1 or c2.
        # Let U be the set of such words.
        # We need c3 to cover all w in U.
        
        # Optimization: Only consider pairs where the union of their "unique" coverage is small.
        # Or just iterate all pairs if N is small enough.
        # N ~ 486. Pairs ~ 118k.
        # For each pair, compute U.
        # If U is empty, we can just remove both (handled by Phase 1 if we checked properly? No, Phase 1 only removes one at a time. If removing c1 is valid, it's removed. If not, c1 is essential. If c1 is essential, it covers some critical word w1. If c2 also covers w1, then removing both might be valid if other words are covered by others.
        # Wait, if c1 is essential, it covers w1 with count 1. If we remove c1 and c2, w1 becomes 0 unless c2 also covers w1? No, if c2 covers w1, count[w1] would be 2. So if count[w1]==1, only c1 covers it. So if we remove c1, w1 is uncovered. To fix, we must add a cw that covers w1.
        
        # So for a pair (c1, c2), the words that need to be covered by a new codeword c3 are:
        # U = { w | cover_count[w] == 1 and (w in coverage[c1] or w in coverage[c2]) }
        # Note: If w is in coverage[c1] and coverage[c2], and count[w]==1, that's impossible because c1 and c2 are distinct and both cover w, so count[w] >= 2.
        # So U is the disjoint union of unique critical words of c1 and unique critical words of c2.
        
        # If U is empty, then removing c1 and c2 leaves no word with count 0. But we must check if any word goes from 2 to 1? No, validity requires count >= 1.
        # If we remove c1 and c2, words covered by them decrease count by 1 (or 2 if covered by both).
        # A word w becomes invalid (count 0) iff count[w] == 1 before removal AND w is covered by c1 or c2.
        # If count[w] == 2 and covered by both c1 and c2, it becomes 0? No, 2 - 2 = 0.
        # So U should also include words w where count[w] == 2 and w is covered by BOTH c1 and c2.
        
        # Let's refine U:
        # U = { w | (count[w] == 1 and w in cov(c1) xor w in cov(c2)) OR (count[w] == 2 and w in cov(c1) and w in cov(c2)) }
        # Actually, if w in cov(c1) and w in cov(c2), and count[w] == 1, impossible.
        # If count[w] == 2 and w in cov(c1) and w in cov(c2), then after removal count is 0. So w must be covered by c3.
        # If count[w] == 1 and w in cov(c1) (and not c2), after removal count is 0. So w must be covered by c3.
        
        # So U is the set of words that will have count 0 if we remove c1 and c2.
        # We need to find c3 such that U subset of cov(c3).
        
        # We can iterate through pairs. To speed up, we can precompute for each cw the list of words it covers.
        # For each pair (c1, c2), we can compute U by intersecting/unioning their coverage lists.
        # Since cover_size is small (17), this is fast.
        
        cws_list = list(set_of_cw)
        found_swap = False
        for i in range(len(cws_list)):
            if time.time() > deadline - 5:
                break
            c1 = cws_list[i]
            for j in range(i + 1, len(cws_list)):
                c2 = cws_list[j]
                
                # Compute U
                # We need words w such that:
                # 1. w is in coverage[c1] or coverage[c2]
                # 2. After removing c1 and c2, count[w] becomes 0.
                # This happens if:
                # - count[w] == 1 and w is covered by exactly one of c1, c2.
                # - count[w] == 2 and w is covered by both c1 and c2.
                
                # Efficient check:
                # Iterate through coverage[c1] and coverage[c2].
                # Since lists are short, we can just check.
                
                u_words = set()
                
                # Words covered by c1
                for j1 in range(cover_size):
                    w = coverage[c1, j1]
                    if cover_count[w] == 1:
                        # Check if c2 covers w
                        # We can check if w is in coverage[c2]
                        # Since cover_size is 17, linear search is fine
                        if w not in coverage[c2]:
                            u_words.add(w)
                    elif cover_count[w] == 2:
                        if w in coverage[c2]:
                            u_words.add(w)
                
                # Words covered by c2 but not c1
                for j2 in range(cover_size):
                    w = coverage[c2, j2]
                    if w in coverage[c1]:
                        continue # Already handled or count >= 2
                    if cover_count[w] == 1:
                        u_words.add(w)
                    # If count[w] == 2 and w in cov(c2) but not c1, then only c2 covers it among the pair?
                    # No, count[w] is global. If w not in c1, and count[w]==2, then removing c2 makes it 1. Still valid.
                    # So we only care if it becomes 0.
                    # It becomes 0 only if count[w] == 1 (and only c2 covers it in the pair? No, if count[w]==1, only one cw in the whole set covers it. If c2 covers it, c1 cannot).
                    # So the logic holds.
                
                if not u_words:
                    # We can remove both!
                    # This is a "double removal".
                    set_of_cw.remove(c1)
                    set_of_cw.remove(c2)
                    for j1 in range(cover_size):
                        w = coverage[c1, j1]
                        cover_count[w] -= 1
                    for j2 in range(cover_size):
                        w = coverage[c2, j2]
                        cover_count[w] -= 1
                    improved = True
                    found_swap = True
                    break
                
                # Check if there is a c3 covering all words in u_words
                # u_words size is small (<= 17 + 17 = 34, usually much less)
                # Pick the first word w0 in u_words.
                # Candidates for c3 must be in covers_word[w0].
                # For each candidate c3, check if it covers all w in u_words.
                
                w0 = next(iter(u_words))
                candidates_c3 = covers_word[w0]
                
                valid_c3 = None
                for k in range(cover_size):
                    c3 = candidates_c3[k]
                    if c3 == c1 or c3 == c2:
                        continue
                    if c3 in set_of_cw:
                        continue # We are looking for a replacement, usually new ones are better, but re-adding existing is allowed? No, c3 must not be c1 or c2. If c3 is in set_of_cw, it's already there. We are removing c1, c2 and adding c3. If c3 is already in set, we just remove c1, c2. But we checked u_words is non-empty, so we need c3 to cover them. If c3 is already in set, it's already counted in cover_count.
                        # Wait, if c3 is in set_of_cw, then cover_count already includes c3.
                        # If we remove c1, c2, and keep c3, the count for w in u_words decreases by 1 or 2.
                        # We need the final count to be >= 1.
                        # My logic for u_words was: words that become 0.
                        # If c3 is already in set, it covers w. So w is covered by c3.
                        # Does that help?
                        # If w is in u_words, it means count[w] becomes 0 after removing c1, c2.
                        # This implies NO other codeword in set_of_cw covers w.
                        # So c3 (if in set_of_cw) cannot cover w.
                        # Therefore, c3 must NOT be in set_of_cw.
                    
                    # Check if c3 covers all w in u_words
                    covers_all = True
                    for w in u_words:
                        # Check if w in coverage[c3]
                        if w not in coverage[c3]:
                            covers_all = False
                            break
                    if covers_all:
                        valid_c3 = c3
                        break
                
                if valid_c3 is not None:
                    # Perform swap
                    set_of_cw.remove(c1)
                    set_of_cw.remove(c2)
                    set_of_cw.add(valid_c3)
                    
                    # Update cover_count
                    for j1 in range(cover_size):
                        w = coverage[c1, j1]
                        cover_count[w] -= 1
                    for j2 in range(cover_size):
                        w = coverage[c2, j2]
                        cover_count[w] -= 1
                    for j3 in range(cover_size):
                        w = coverage[valid_c3, j3]
                        cover_count[w] += 1
                        
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                break
        
        if found_swap:
            continue

        # If no improvements found in this round, we might be stuck.
        # We could try triple swaps, but let's see if single/pair is enough.
        # Given the time, we break if no improvement.
        break

    result = [idx_to_digits(cw) for cw in set_of_cw]
    
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
            
    return unique_result
