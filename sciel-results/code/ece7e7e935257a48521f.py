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
    # Ball[w] contains w and all words at Hamming distance 1 from w
    # Size is 1 + 2*n = 17
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
            # counts[c] = number of uncovered words in balls[c]
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
        # Phase 1: Remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: Swap / Replace
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                # Check if removable again
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Find critical words: covered ONLY by c
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Find candidates c' such that critical_words are all in balls[c']
                # Start with the ball of the first critical word
                candidates = set(balls[critical_words[0]].tolist())
                
                # Intersect with balls of other critical words
                for cw in critical_words[1:]:
                    candidates.intersection_update(balls[cw].tolist())
                    if not candidates:
                        break
                
                # Remove c itself
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
            
        # Phase 3: 2-Exchange (Remove 2, Add 1)
        if len(selected) > 450 and time.time() < deadline - 8.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                improved_2ex = False
                new_selected = list(selected)
                
                # Try random pairs
                indices = list(range(len(new_selected)))
                random.shuffle(indices)
                
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if time.time() > deadline - 5.0:
                            break
                            
                        c1 = new_selected[indices[i]]
                        c2 = new_selected[indices[j]]
                        
                        # Temporarily remove
                        cover_count[balls[c1]] -= 1
                        cover_count[balls[c2]] -= 1
                        
                        uncovered_after = (cover_count == 0)
                        
                        if np.any(uncovered_after):
                            uncovered_words = np.where(uncovered_after)[0]
                            
                            # Find c3 that covers all uncovered words
                            # Intersect balls of all uncovered words
                            candidates = set(balls[uncovered_words[0]].tolist())
                            for uw in uncovered_words[1:]:
                                candidates.intersection_update(balls[uw].tolist())
                                if not candidates:
                                    break
                            
                            candidates.discard(c1)
                            candidates.discard(c2)
                            
                            if candidates:
                                c3 = random.choice(list(candidates))
                                cover_count[balls[c3]] += 1
                                new_selected.remove(c1)
                                new_selected.remove(c2)
                                new_selected.append(c3)
                                improved_2ex = True
                                break
                            else:
                                # Revert
                                cover_count[balls[c1]] += 1
                                cover_count[balls[c2]] += 1
                        else:
                            # Both can be removed
                            new_selected.remove(c1)
                            new_selected.remove(c2)
                            improved_2ex = True
                            break
                            
                        if not improved_2ex:
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                        
                    if improved_2ex:
                        break
                
                selected = new_selected
                
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break
            if best_size <= 486:
                # Known best
                # We can break to ensure we return quickly
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
