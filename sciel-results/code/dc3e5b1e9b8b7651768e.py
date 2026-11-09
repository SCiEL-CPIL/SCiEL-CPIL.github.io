"""Covering code K_3(8,1) via direct-sum init + greedy removal + critical-word local search."""
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Fallback for non-target parameters
    if not (q == 3 and n == 8 and radius == 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n  # 6561
    
    # Helper to convert int to base-q digits (MSB first)
    def int_to_digits(num, length):
        digits = [0] * length
        for i in range(length - 1, -1, -1):
            digits[i] = num % q
            num //= q
        return digits

    # Helper to convert base-q digits to int
    def digits_to_int(digits):
        val = 0
        for d in digits:
            val = val * q + d
        return val

    # Precompute balls as lists of indices
    balls = []
    for i in range(N):
        digits = int_to_digits(i, n)
        ball = [i]
        for k in range(n):
            orig_val = digits[k]
            for v in range(q):
                if v == orig_val:
                    continue
                new_digits = digits.copy()
                new_digits[k] = v
                j = digits_to_int(new_digits)
                ball.append(j)
        balls.append(ball)

    # Build incidence matrix M: M[i, j] = 1 if word j is in ball of word i
    M = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in balls[i]:
            M[i, j] = 1

    # Initial seed: tetracode direct sum (486 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(tuple(h + list(tail)))
    
    # Convert to integer indices
    codewords = [digits_to_int(list(cw)) for cw in initial_codewords]
    codewords_set = set(codewords)
    
    # Verify initial cover is valid (it should be)
    covered = np.zeros(N, dtype=np.uint8)
    for c in codewords:
        covered[M[c] == 1] = 1
    if not covered.all():
        # Should not happen, but add greedy completion if needed
        uncovered = 1 - covered
        while uncovered.any():
            if time.time() - start_time > deadline - 10:
                break
            scores = M.dot(uncovered)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            codewords.append(best_idx)
            codewords_set.add(best_idx)
            covered[M[best_idx] == 1] = 1
            uncovered = 1 - covered

    # Phase 1: Greedy removal - remove redundant codewords
    def try_remove_redundant(cw_list, cw_set, M, balls, deadline):
        """Remove any codeword whose ball is fully covered by others."""
        changed = True
        while changed:
            if time.time() - start_time > deadline - 15:
                break
            changed = False
            counts = np.zeros(N, dtype=np.int32)
            for c in cw_list:
                counts[balls[c]] += 1
            
            new_list = []
            for c in cw_list:
                counts[balls[c]] -= 1
                if np.min(counts) >= 1:
                    # Redundant
                    cw_set.discard(c)
                    changed = True
                else:
                    counts[balls[c]] += 1
                    new_list.append(c)
            cw_list = new_list
        return cw_list

    codewords = try_remove_redundant(codewords, codewords_set, M, balls, deadline)

    # Phase 2: Local search - try to reduce count by replacing codewords
    # Strategy: For each codeword, try replacing it with another word such that
    # the new code still covers everything and we can then remove another codeword.
    # More practical: try 1-for-1 swaps that allow removal, or try to find
    # codewords that can be replaced to enable removal of a pair.
    
    # Simpler local search: try removing each codeword and see if we can add
    # one codeword that covers all the now-uncovered words.
    
    def local_search(cw_list, cw_set, M, balls, deadline, max_iters=500):
        """Try to reduce code size by 1 via swap operations."""
        current_size = len(cw_list)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            if time.time() - start_time > deadline - 10:
                break
            improved = False
            iters += 1
            
            # Try each codeword for removal
            for i, c in enumerate(list(cw_list)):
                if time.time() - start_time > deadline - 10:
                    break
                
                # Compute what becomes uncovered if we remove c
                counts = np.zeros(N, dtype=np.int32)
                for other in cw_list:
                    if other != c:
                        counts[balls[other]] += 1
                
                uncovered = counts < 1
                if not uncovered.any():
                    # c is redundant, remove it
                    cw_list.remove(c)
                    cw_set.discard(c)
                    improved = True
                    break
                
                # Find if there's a single word that covers all uncovered
                # uncovered words
                # The union of balls that cover all uncovered words
                # We need a word w such that ball[w] covers all uncovered
                # i.e., M[w, uncovered] are all 1
                
                # For each candidate w not in cw_list, check if it covers all uncovered
                # This is expensive, so we limit candidates
                # The uncovered words must all be in ball[w]
                # So w must be within distance 1 of each uncovered word
                
                # Get the set of uncovered words
                uncov_indices = np.where(uncovered)[0]
                
                # For each uncovered word, the candidates that cover it are
                # the words in its ball. We need intersection of all such balls.
                # Start with first uncovered word's ball
                if len(uncov_indices) == 0:
                    continue
                    
                # Candidates that cover uncov_indices[0]
                # M[:, uncov_indices[0]] == 1 gives rows that cover this word
                # But we want columns: which words w have M[w, uncov_indices[0]] == 1
                # That's M[:, uncov_indices[0]]
                
                candidate_mask = M[:, uncov_indices[0]].astype(bool)
                
                for ui in uncov_indices[1:]:
                    candidate_mask &= M[:, ui].astype(bool)
                    if not candidate_mask.any():
                        break
                
                if candidate_mask.any():
                    # Find candidates not in current code
                    in_code = np.zeros(N, dtype=bool)
                    for cw in cw_list:
                        in_code[cw] = True
                    
                    candidate_mask &= ~in_code
                    
                    if candidate_mask.any():
                        # Pick the candidate that covers the most total words
                        # (to maximize future flexibility)
                        candidates = np.where(candidate_mask)[0]
                        # Score by total coverage
                        best_cand = None
                        best_score = -1
                        for cand in candidates:
                            score = np.sum(M[cand])
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                        
                        if best_cand is not None:
                            # Perform the swap: remove c, add best_cand
                            cw_list.remove(c)
                            cw_set.discard(c)
                            cw_list.append(best_cand)
                            cw_set.add(best_cand)
                            improved = True
                            break
        
        return cw_list

    codewords = local_search(codewords, codewords_set, M, balls, deadline)

    # Phase 3: More aggressive local search - try pairwise operations
    # Try to find two codewords that can be replaced by one
    def pairwise_local_search(cw_list, cw_set, M, balls, deadline):
        """Try to replace two codewords with one."""
        improved = True
        iters = 0
        max_iters = 200
        
        while improved and iters < max_iters:
            if time.time() - start_time > deadline - 5:
                break
            improved = False
            iters += 1
            
            # Try each pair
            for i in range(len(cw_list)):
                if time.time() - start_time > deadline - 5:
                    break
                for j in range(i + 1, len(cw_list)):
                    if time.time() - start_time > deadline - 5:
                        break
                    
                    c1 = cw_list[i]
                    c2 = cw_list[j]
                    
                    # Compute what becomes uncovered if we remove both
                    counts = np.zeros(N, dtype=np.int32)
                    for other in cw_list:
                        if other != c1 and other != c2:
                            counts[balls[other]] += 1
                    
                    uncovered = counts < 1
                    if not uncovered.any():
                        # Both redundant, remove both
                        cw_list.remove(c1)
                        cw_list.remove(c2)
                        cw_set.discard(c1)
                        cw_set.discard(c2)
                        improved = True
                        break
                    
                    uncov_indices = np.where(uncovered)[0]
                    if len(uncov_indices) == 0:
                        continue
                    
                    # Find if a single word covers all uncovered
                    candidate_mask = M[:, uncov_indices[0]].astype(bool)
                    for ui in uncov_indices[1:]:
                        candidate_mask &= M[:, ui].astype(bool)
                        if not candidate_mask.any():
                            break
                    
                    if candidate_mask.any():
                        in_code = np.zeros(N, dtype=bool)
                        for cw in cw_list:
                            in_code[cw] = True
                        candidate_mask &= ~in_code
                        
                        if candidate_mask.any():
                            candidates = np.where(candidate_mask)[0]
                            # Pick best
                            best_cand = None
                            best_score = -1
                            for cand in candidates:
                                score = np.sum(M[cand])
                                if score > best_score:
                                    best_score = score
                                    best_cand = cand
                            
                            if best_cand is not None:
                                # Replace c1 and c2 with best_cand
                                cw_list.remove(c1)
                                cw_list.remove(c2)
                                cw_set.discard(c1)
                                cw_set.discard(c2)
                                cw_list.append(best_cand)
                                cw_set.add(best_cand)
                                improved = True
                                break
                
                if improved:
                    break
        
        return cw_list

    codewords = pairwise_local_search(codewords, codewords_set, M, balls, deadline)

    # Final greedy removal pass
    codewords = try_remove_redundant(codewords, codewords_set, M, balls, deadline)

    # Verify validity
    covered = np.zeros(N, dtype=np.uint8)
    for c in codewords:
        covered[M[c] == 1] = 1
    
    if not covered.all():
        # Emergency: add missing coverage
        uncovered = 1 - covered
        while uncovered.any():
            if time.time() - start_time > deadline - 1:
                break
            scores = M.dot(uncovered)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            if best_idx not in codewords_set:
                codewords.append(best_idx)
                codewords_set.add(best_idx)
            covered[M[best_idx] == 1] = 1
            uncovered = 1 - covered

    # Convert back to lists of integers
    result = []
    seen = set()
    for c in codewords:
        digits = int_to_digits(c, n)
        t = tuple(digits)
        if t not in seen:
            seen.add(t)
            result.append(digits)
    
    return result
