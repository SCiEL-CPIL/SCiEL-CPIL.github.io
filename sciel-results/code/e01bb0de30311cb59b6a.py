import time
import numpy as np
import random
import itertools
from functools import lru_cache

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
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
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
    
    # Precompute frozensets for fast intersection
    ball_sets = [frozenset(balls[w].tolist()) for w in range(total)]
    
    best_size = total
    best_set = None
    
    # Run multiple restarts
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break
            # Count coverage for each candidate among uncovered words
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid due to timeout
        
        # 2. Local Search
        # Phase 1: Remove redundant codewords (multiple passes)
        for _pass in range(3):
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            if not changed:
                break
            
        # Phase 2: 1-Swap (Remove 1, Add 1) using precomputed sets
        improved = True
        while improved and time.time() < deadline - 5.0:
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

                # Use precomputed frozensets for fast intersection
                candidates = ball_sets[critical_words[0]]
                for cw in critical_words[1:]:
                    candidates = candidates & ball_sets[cw]
                    if not candidates:
                        break
                
                if c in candidates:
                    candidates = candidates - {c}
                
                if candidates:
                    c_prime = random.choice(tuple(candidates))
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1) with Random Sampling
        if len(selected) > 486 and time.time() < deadline - 8.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                improved_2ex = False
                num_trials = min(8000, len(selected) * (len(selected) - 1) // 2)
                
                for _ in range(num_trials):
                    if time.time() > deadline - 5.0:
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
                        
                        # Find c3 that covers all uncovered words using precomputed sets
                        candidates = ball_sets[uncovered_words[0]]
                        for uw in uncovered_words[1:]:
                            candidates = candidates & ball_sets[uw]
                            if not candidates:
                                break
                        
                        if c1 in candidates:
                            candidates = candidates - {c1}
                        if c2 in candidates:
                            candidates = candidates - {c2}
                        
                        if candidates:
                            c3 = random.choice(tuple(candidates))
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
        
        # Phase 4: Remove 1, Add 2 (can reduce size if two new codewords cover more)
        # Only try when we have room in budget
        if len(selected) > 490 and time.time() < deadline - 10.0:
            improved_12 = True
            while improved_12 and time.time() < deadline - 8.0:
                improved_12 = False
                num_trials_12 = min(3000, len(selected))
                
                for _ in range(num_trials_12):
                    if time.time() > deadline - 8.0:
                        break
                    
                    idx_rem = random.randint(0, len(selected) - 1)
                    c_rem = selected[idx_rem]
                    
                    # Temporarily remove
                    cover_count[balls[c_rem]] -= 1
                    uncovered_after = (cover_count == 0)
                    
                    if not np.any(uncovered_after):
                        # Can simply remove
                        new_selected = [c for c in selected if c != c_rem]
                        selected = new_selected
                        improved_12 = True
                        break
                    else:
                        uncovered_words = np.where(uncovered_after)[0]
                        
                        # Try to find two codewords that together cover all uncovered
                        # First, find candidates that cover at least one uncovered word
                        # Use a greedy approach: pick best single candidate, then find second
                        best_single = None
                        best_single_count = 0
                        for uw in uncovered_words[:5]:  # Limit to first 5 for speed
                            for cand in ball_sets[uw]:
                                if cand == c_rem:
                                    continue
                                # Count how many uncovered words this candidate covers
                                cnt = sum(1 for uw2 in uncovered_words if uw2 in ball_sets[cand])
                                if cnt > best_single_count:
                                    best_single_count = cnt
                                    best_single = cand
                        
                        if best_single is not None and best_single_count < len(uncovered_words):
                            # Find second candidate to cover remaining
                            remaining = [uw for uw in uncovered_words if uw not in ball_sets[best_single]]
                            if remaining:
                                best_second = None
                                best_second_count = 0
                                for uw in remaining[:3]:
                                    for cand in ball_sets[uw]:
                                        if cand == c_rem or cand == best_single:
                                            continue
                                        cnt = sum(1 for uw2 in remaining if uw2 in ball_sets[cand])
                                        if cnt > best_second_count:
                                            best_second_count = cnt
                                            best_second = cand
                                
                                if best_second is not None:
                                    # Verify both together cover all uncovered
                                    all_covered = all(
                                        uw in ball_sets[best_single] or uw in ball_sets[best_second]
                                        for uw in uncovered_words
                                    )
                                    if all_covered:
                                        cover_count[balls[best_single]] += 1
                                        cover_count[balls[best_second]] += 1
                                        new_selected = [c for c in selected if c != c_rem]
                                        new_selected.append(best_single)
                                        new_selected.append(best_second)
                                        selected = new_selected
                                        improved_12 = True
                                        break
                        
                        # Revert
                        cover_count[balls[c_rem]] += 1
                        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break
            if best_size <= 486:
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback
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
