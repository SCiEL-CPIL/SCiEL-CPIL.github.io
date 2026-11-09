import numpy as np
import time
import itertools

def construct(q, n, radius):
    # Fallback for non-target parameters
    if not (q == 3 and n == 8 and radius == 1):
        # Use the seed logic for other cases
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Handle small n generically if needed, though task is fixed
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 10.0
    
    # Precompute balls (coverage sets) using vectorized approach
    # M is the incidence matrix
    M = np.zeros((N, N), dtype=np.uint8)
    
    # Precompute digit representations for all numbers
    # digits[i] = base-q representation of i, length n
    digits_arr = np.zeros((N, n), dtype=np.int32)
    for i in range(N):
        num = i
        for k in range(n):
            digits_arr[i, k] = num % q
            num //= q
    
    # Build M vectorized: for each position k and each value v != digits[i,k]
    # the neighbor index is computed
    # For position k, changing digit k from val to new_val changes index by (new_val - val) * q^k
    powers = np.array([q ** k for k in range(n)], dtype=np.int64)
    
    for k in range(n):
        for v in range(q):
            # For all i where digits_arr[i, k] != v
            mask = (digits_arr[:, k] != v)
            if not np.any(mask):
                continue
            # For these i, the neighbor j = i + (v - digits_arr[i,k]) * q^k
            delta = (v - digits_arr[mask, k]) * powers[k]
            j_indices = (np.arange(N)[mask] + delta).astype(np.int64)
            i_indices = np.arange(N)[mask]
            M[i_indices, j_indices] = 1
    
    # Set diagonal
    np.fill_diagonal(M, 1)
    
    # balls[i] is a list of indices j covered by codeword i
    # Store as list of arrays for faster indexing
    balls = []
    for i in range(N):
        balls.append(np.where(M[i] == 1)[0])
    
    # Phase 1: Greedy Construction
    uncovered = np.ones(N, dtype=np.uint8)
    codewords = []
    
    while uncovered.any():
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        # Calculate how many uncovered words each candidate covers
        scores = M.dot(uncovered)
        
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
            break
            
        codewords.append(best_idx)
        uncovered[M[best_idx] == 1] = 0

    # Phase 2: Greedy Removal (multiple passes)
    counts = np.zeros(N, dtype=np.int32)
    for c in codewords:
        counts[balls[c]] += 1

    for _pass in range(5):
        if time.time() - start_time > time_limit - safety_margin:
            break
        changed = True
        while changed and (time.time() - start_time < time_limit - safety_margin):
            changed = False
            new_codewords = []
            for c in codewords:
                mask = balls[c]
                counts[mask] -= 1
                if np.min(counts) >= 1:
                    changed = True
                else:
                    counts[mask] += 1
                    new_codewords.append(c)
            codewords = new_codewords

    # Phase 3: Local Search (Swap-based optimization)
    if time.time() - start_time < time_limit - safety_margin:
        counts = np.zeros(N, dtype=np.int32)
        for c in codewords:
            counts[balls[c]] += 1
        
        cw_set = set(codewords)
        
        # Precompute total coverage for each codeword for faster scoring
        total_cov = M.sum(axis=1)
        
        # Precompute coverage count for each position by each codeword
        # For swap scoring, we want to maximize the number of newly covered positions
        improved = True
        while improved and (time.time() - start_time < time_limit - safety_margin):
            improved = False
            new_codewords = list(codewords)
            
            for i, c in enumerate(codewords):
                if time.time() - start_time > time_limit - safety_margin:
                    break
                
                counts[balls[c]] -= 1
                min_count = np.min(counts)
                
                if min_count >= 1:
                    new_codewords.pop(i)
                    cw_set.discard(c)
                    improved = True
                    continue
                else:
                    critical_mask = (counts == 1)
                    critical_indices = np.where(critical_mask)[0]
                    
                    if len(critical_indices) > 0:
                        # Only consider candidates that cover all critical positions
                        crit_coverage = M[:, critical_indices].sum(axis=1)
                        all_covered = crit_coverage == len(critical_indices)
                        
                        if np.any(all_covered):
                            candidate_indices = np.where(all_covered)[0]
                            # Exclude current codewords, pick best by total coverage
                            valid_candidates = [d for d in candidate_indices if d not in cw_set]
                            
                            if valid_candidates:
                                # Score by how many additional positions the candidate covers
                                # beyond what is already covered
                                best_d = max(valid_candidates, key=lambda d: total_cov[d])
                                
                                new_codewords[i] = best_d
                                cw_set.discard(c)
                                cw_set.add(best_d)
                                counts[balls[c]] += 1
                                counts[balls[best_d]] += 1
                                improved = True
                                continue
                    else:
                        new_codewords.pop(i)
                        cw_set.discard(c)
                        improved = True
                        continue
                
                counts[balls[c]] += 1
            
            codewords = new_codewords
        
        # Phase 3b: Additional removal passes after swaps
        for _rem_pass in range(3):
            if time.time() - start_time > time_limit - safety_margin:
                break
            counts = np.zeros(N, dtype=np.int32)
            for c in codewords:
                counts[balls[c]] += 1
            changed = True
            while changed and (time.time() - start_time < time_limit - safety_margin):
                changed = False
                new_codewords = []
                for c in codewords:
                    mask = balls[c]
                    counts[mask] -= 1
                    if np.min(counts) >= 1:
                        changed = True
                    else:
                        counts[mask] += 1
                        new_codewords.append(c)
                codewords = new_codewords
        
        # Phase 3c: Second round of swaps
        if time.time() - start_time < time_limit - safety_margin:
            counts = np.zeros(N, dtype=np.int32)
            for c in codewords:
                counts[balls[c]] += 1
            cw_set = set(codewords)
            improved = True
            while improved and (time.time() - start_time < time_limit - safety_margin):
                improved = False
                new_codewords = list(codewords)
                
                for i, c in enumerate(codewords):
                    if time.time() - start_time > time_limit - safety_margin:
                        break
                    
                    counts[balls[c]] -= 1
                    min_count = np.min(counts)
                    
                    if min_count >= 1:
                        new_codewords.pop(i)
                        cw_set.discard(c)
                        improved = True
                        continue
                    else:
                        critical_mask = (counts == 1)
                        critical_indices = np.where(critical_mask)[0]
                        
                        if len(critical_indices) > 0:
                            crit_coverage = M[:, critical_indices].sum(axis=1)
                            all_covered = crit_coverage == len(critical_indices)
                            
                            if np.any(all_covered):
                                candidate_indices = np.where(all_covered)[0]
                                valid_candidates = [d for d in candidate_indices if d not in cw_set]
                                
                                if valid_candidates:
                                    best_d = max(valid_candidates, key=lambda d: total_cov[d])
                                    
                                    new_codewords[i] = best_d
                                    cw_set.discard(c)
                                    cw_set.add(best_d)
                                    counts[balls[c]] += 1
                                    counts[balls[best_d]] += 1
                                    improved = True
                                    continue
                        else:
                            new_codewords.pop(i)
                            cw_set.discard(c)
                            improved = True
                            continue
                    
                    counts[balls[c]] += 1
                
                codewords = new_codewords
    
    # Convert back to lists of integers
    def int_to_digits(num, length):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % q
            num //= q
        return digits

    result = []
    for c in codewords:
        digits = int_to_digits(c, n)
        result.append(digits[::-1])

    # Verify coverage (safety check)
    if time.time() - start_time < time_limit - safety_margin:
        covered = np.zeros(N, dtype=np.uint8)
        for c in codewords:
            covered[balls[c]] = 1
        if not covered.all():
            # Fallback to full space if verification fails (shouldn't happen)
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    return result
