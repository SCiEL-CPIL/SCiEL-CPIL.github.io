import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    # balls[w] is a list of 17 indices: w itself and its 16 neighbors
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # Precompute ball membership as a boolean matrix for fast subset checks
    # ball_membership[w, v] = True if v is in balls[w]
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True
    
    best_size = total
    best_set = None
    
    # Run multiple restarts with different strategies
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 15.0:
                break
            # Count coverage for each candidate among uncovered words
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best with some noise
            if np.random.random() < 0.3 and len(candidates) > 1:
                # Add noise: pick from top 30% of candidates
                top_fraction = max(1, len(candidates) // 3)
                subset = candidates[:top_fraction]
                best_idx = subset[np.random.randint(len(subset))]
            else:
                best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue  # Invalid due to timeout
        
        # 2. Local Search
        # Phase 1: Remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: 1-Swap (Remove 1, Add 1)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Find candidates that cover all critical words
                candidate_mask = np.ones(total, dtype=bool)
                for cw in critical_words:
                    candidate_mask &= ball_membership[:, cw]
                
                valid_candidates = np.where(candidate_mask)[0]
                valid_candidates = valid_candidates[valid_candidates != c]
                
                if len(valid_candidates) > 0:
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1) with Random Sampling
        # To avoid O(N^2) full scan, we sample pairs
        if len(selected) > 486 and time.time() < deadline - 15.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 10.0:
                improved_2ex = False
                # Sample a subset of pairs to try
                num_trials = min(3000, len(selected) * (len(selected) - 1) // 2)
                
                for _ in range(num_trials):
                    if time.time() > deadline - 10.0:
                        break
                    
                    idx1 = random.randint(0, len(selected) - 1)
                    idx2 = random.randint(0, len(selected) - 1)
                    if idx1 == idx2:
                        continue
                        
                    c1 = selected[idx1]
                    c2 = selected[idx2]
                    
                    # Temporarily remove
                    cover_count[balls[c1]] -= 1
                    cover_count[balls[c2]] -= 1
                    
                    uncovered_after = (cover_count == 0)
                    
                    if np.any(uncovered_after):
                        uncovered_words = np.where(uncovered_after)[0]
                        
                        # Find c3 that covers all uncovered words
                        candidate_mask = np.ones(total, dtype=bool)
                        for uw in uncovered_words:
                            candidate_mask &= ball_membership[:, uw]
                        
                        valid_candidates = np.where(candidate_mask)[0]
                        valid_candidates = valid_candidates[(valid_candidates != c1) & (valid_candidates != c2)]
                        
                        if len(valid_candidates) > 0:
                            c3 = valid_candidates[np.random.randint(len(valid_candidates))]
                            cover_count[balls[c3]] += 1
                            # Remove c1, c2 from selected
                            new_selected = []
                            for c in selected:
                                if c != c1 and c != c2:
                                    new_selected.append(c)
                            new_selected.append(c3)
                            selected = new_selected
                            improved_2ex = True
                            break
                        else:
                            # Revert
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        # Both can be removed
                        new_selected = []
                        for c in selected:
                            if c != c1 and c != c2:
                                new_selected.append(c)
                        selected = new_selected
                        improved_2ex = True
                        break
                        
                    if not improved_2ex:
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            if best_size <= 403:
                break
            if best_size <= 486:
                # Do one more polish pass
                break
        
        # If we've found a good solution, do a final polish
        if best_size <= 490 and time.time() < deadline - 20.0:
            # Reconstruct cover_count for best_set
            cc = np.zeros(total, dtype=np.int32)
            for c in best_set:
                cc[balls[c]] += 1
            
            # Final polish: remove redundant
            changed = True
            while changed and time.time() < deadline - 10.0:
                changed = False
                new_sel = []
                for c in best_set:
                    if np.min(cc[balls[c]]) > 1:
                        cc[balls[c]] -= 1
                        changed = True
                    else:
                        new_sel.append(c)
                best_set = new_sel
            
            # Final swap pass
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                new_sel = []
                for c in best_set:
                    if np.min(cc[balls[c]]) > 1:
                        cc[balls[c]] -= 1
                        improved = True
                        continue
                    
                    critical_mask = (cc[balls[c]] == 1)
                    critical_words = balls[c][critical_mask]
                    
                    if len(critical_words) == 0:
                        cc[balls[c]] -= 1
                        improved = True
                        continue

                    candidate_mask = np.ones(total, dtype=bool)
                    for cw in critical_words:
                        candidate_mask &= ball_membership[:, cw]
                    
                    valid_candidates = np.where(candidate_mask)[0]
                    valid_candidates = valid_candidates[valid_candidates != c]
                    
                    if len(valid_candidates) > 0:
                        c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                        cc[balls[c]] -= 1
                        cc[balls[c_prime]] += 1
                        new_sel.append(c_prime)
                        improved = True
                    else:
                        new_sel.append(c)
                
                best_set = new_sel
            
            if len(best_set) < best_size:
                best_size = len(best_set)
                if best_size <= 403:
                    break
                if best_size <= 486:
                    break

    # Convert best_set to output format
    if best_set is None:
        # Fallback: simple greedy
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
