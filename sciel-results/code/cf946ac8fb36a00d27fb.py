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
    
    # Precompute frozensets for fast intersection in local search
    ball_sets = [frozenset(balls[w].tolist()) for w in range(total)]
    
    best_size = total
    best_set = None
    
    # Precompute digit representations for Hamming distance
    digit_rep = np.zeros((total, n), dtype=np.int8)
    for w in range(total):
        temp = w
        for i in range(n - 1, -1, -1):
            digit_rep[w, i] = temp % 3
            temp //= 3
    
    def hamming_dist(a, b):
        return np.sum(digit_rep[a] != digit_rep[b])
    
    # Run multiple restarts
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        seed = iteration * 12345 + 67890
        np.random.seed(seed)
        random.seed(seed)
        
        # 1. Randomized Greedy Construction with diversity bias
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
            
            # Diversity bias: prefer candidates far from recently selected
            if selected and len(candidates) > 1:
                recent = selected[-min(15, len(selected)):]
                min_dists = np.array([min(hamming_dist(cand, s) for s in recent) for cand in candidates])
                max_d = np.max(min_dists)
                # Soft-max weighting
                weights_arr = np.exp((min_dists - max_d) / 1.5)
                weights_arr /= weights_arr.sum()
                best_idx = candidates[np.random.choice(len(candidates), p=weights_arr)]
            else:
                best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue  # Invalid due to timeout
        
        # 2. Local Search
        # Phase 1: Remove redundant codewords (multiple passes)
        for _pass in range(5):
            if time.time() > deadline - 5.0:
                break
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
        
        # Phase 2: 1-Swap (Remove 1, Add 1)
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

                # Use frozenset intersection for speed
                candidates = set(ball_sets[critical_words[0]])
                for cw in critical_words[1:]:
                    candidates = candidates & ball_sets[cw]
                    if not candidates:
                        break
                
                candidates.discard(c)
                
                if candidates:
                    c_prime = random.choice(list(candidates))
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
                num_trials = min(10000, len(selected) * (len(selected) - 1) // 2)
                
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
                        
                        if len(uncovered_words) <= 6:
                            candidates = set(ball_sets[uncovered_words[0]])
                            for uw in uncovered_words[1:]:
                                candidates = candidates & ball_sets[uw]
                                if not candidates:
                                    break
                            
                            candidates.discard(c1)
                            candidates.discard(c2)
                            
                            if candidates:
                                c3 = random.choice(list(candidates))
                                cover_count[balls[c3]] += 1
                                # Remove c1, c2 from selected
                                new_selected = [c for c in selected if c != c1 and c != c2]
                                new_selected.append(c3)
                                selected = new_selected
                                improved_2ex = True
                                break
                            else:
                                cover_count[balls[c1]] += 1
                                cover_count[balls[c2]] += 1
                        else:
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        # Both can be removed
                        new_selected = [c for c in selected if c != c1 and c != c2]
                        selected = new_selected
                        improved_2ex = True
                        break
                        
                    if not improved_2ex:
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1
        
        # Phase 4: Final polish - try more aggressive 2-exchanges and swaps
        if time.time() < deadline - 3.0:
            for _polish in range(3):
                if time.time() > deadline - 3.0:
                    break
                # Another round of removals
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
                
                # Another round of 1-swaps
                improved = True
                while improved and time.time() < deadline - 3.0:
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
                        candidates = set(ball_sets[critical_words[0]])
                        for cw in critical_words[1:]:
                            candidates = candidates & ball_sets[cw]
                            if not candidates:
                                break
                        candidates.discard(c)
                        if candidates:
                            c_prime = random.choice(list(candidates))
                            cover_count[balls[c]] -= 1
                            cover_count[balls[c_prime]] += 1
                            new_selected.append(c_prime)
                            improved = True
                        else:
                            new_selected.append(c)
                    selected = new_selected
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
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
