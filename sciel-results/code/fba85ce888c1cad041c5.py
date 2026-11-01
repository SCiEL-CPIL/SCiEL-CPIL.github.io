import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, ball_size), dtype=np.int32)
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
    
    # Build inverse mapping (list of codewords covering each word)
    inv_lists = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv_lists[w].append(c)
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    # Initial Greedy Construction
    def greedy_init():
        selected = []
        selected_set = set()
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total
        
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            cand_count[c] = ball_size
            
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0 and time.time() < deadline - 10.0:
            # Find best candidate: max uncovered neighbors
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)
            if max_count <= 0:
                break
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            active[best_idx] = False
            
            # Update coverage and candidate counts
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1
                
        return selected, selected_set, cover_count, uncovered_count

    # Main Loop: Greedy + SA
    while time.time() < deadline - 5.0:
        selected, selected_set, cover_count, uncovered_count = greedy_init()
        
        if uncovered_count > 0:
            continue
            
        # Prune initial solution (remove redundant codewords)
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected_set.discard(c)
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        alpha = 0.99995
        
        def get_critical_count(cover_count):
            return np.sum(cover_count == 1)
            
        current_critical = get_critical_count(cover_count)
        stall_count = 0
        
        while time.time() < deadline - 2.0 and T > T_min:
            move_made = False
            
            if len(selected) == 0:
                break
                
            c_old = selected[random.randint(0, len(selected) - 1)]
            
            # Determine words that would become uncovered if c_old is removed
            ball_c_old = balls[c_old]
            critical_words = ball_c_old[cover_count[ball_c_old] == 1]
            
            if len(critical_words) == 0:
                # Can remove
                cover_count[ball_c_old] -= 1
                selected.remove(c_old)
                selected_set.discard(c_old)
                current_critical = get_critical_count(cover_count)
                move_made = True
                stall_count = 0
            else:
                # Need to swap c_old with c_new
                # c_new must cover all critical_words
                # Find candidates: intersection of balls of critical_words
                # Optimization: Use the smallest list to start intersection
                candidates = None
                if len(critical_words) == 1:
                    w = critical_words[0]
                    candidates = [c for c in inv_lists[w] if c != c_old and c not in selected_set]
                elif len(critical_words) == 2:
                    w1, w2 = critical_words[0], critical_words[1]
                    # Intersect
                    set1 = set(inv_lists[w1])
                    set2 = set(inv_lists[w2])
                    candidates = [c for c in set1.intersection(set2) if c != c_old and c not in selected_set]
                elif len(critical_words) <= 4:
                    common = None
                    for w in critical_words:
                        if common is None:
                            common = set(inv_lists[w])
                        else:
                            common.intersection_update(inv_lists[w])
                    if common:
                        candidates = [c for c in common if c != c_old and c not in selected_set]
                    else:
                        candidates = []
                else:
                    candidates = []
                
                if candidates:
                    # Heuristic: Pick the candidate that maximizes redundancy
                    # Score(w) = sum(cover_count[balls[w]])
                    # We want to move coverage to areas that are already well-covered
                    # to potentially make other codewords removable.
                    
                    best_c_new = -1
                    best_score = -1
                    
                    # To speed up, if candidates list is huge, we might sample.
                    # But usually it's small (<100).
                    # Vectorized scoring:
                    # We can't easily vectorize sum(cover_count[balls[w]]) for all w in candidates
                    # without a loop, because balls[w] varies.
                    # However, we can approximate or just loop.
                    
                    for c_new in candidates:
                        # Check validity:
                        # Remove c_old, Add c_new
                        # Words in ball[c_old] \ ball[c_new] must have cover_count > 1
                        # i.e., min(cover_count[ball[c_old] \ ball[c_new]]) >= 2
                        
                        # Quick check:
                        # If c_new is in ball[c_old], intersection is large.
                        # Let's just do the full check for the top few candidates?
                        # Or just pick the one with highest score and check validity?
                        
                        # Let's compute score first
                        score = np.sum(cover_count[balls[c_new]])
                        
                        if score > best_score:
                            # Verify validity
                            # Simulate removal of c_old
                            # If min(cover_count[ball_c_old]) - 1 < 0 for words not in ball[c_new], it's invalid.
                            # Actually, simpler:
                            # new_cover_count = cover_count
                            # new_cover_count[ball_c_old] -= 1
                            # if np.min(new_cover_count[ball_c_old]) < 0: invalid
                            # But we only care about words that LOSE coverage.
                            # Words in ball[c_old] but NOT in ball[c_new] lose coverage.
                            # Words in intersection keep coverage (lose 1, gain 1).
                            # Words only in c_new gain coverage.
                            
                            # Let's check min cover count of words in ball[c_old] that are not in ball[c_new].
                            # This is expensive to compute intersection for every candidate.
                            
                            # Alternative: Just pick the best score candidate and try the swap.
                            # If it fails, try the next best?
                            # For now, let's assume the best score candidate is likely valid
                            # if it covers the critical words (which it does by construction).
                            # The risk is uncovering other words.
                            
                            best_c_new = c_new
                            best_score = score
                    
                    if best_c_new != -1:
                        c_new = best_c_new
                        
                        # Perform Swap
                        # Store old values
                        old_counts_c_old = cover_count[ball_c_old].copy()
                        old_counts_c_new = cover_count[balls[c_new]].copy()
                        
                        # Apply
                        cover_count[ball_c_old] -= 1
                        
                        # Check validity
                        if np.min(cover_count[ball_c_old]) < 0:
                            # Invalid, revert
                            cover_count[ball_c_old] = old_counts_c_old
                            # Try next best candidate? For simplicity, stall.
                            stall_count += 1
                            if stall_count > 50:
                                break
                            continue
                        
                        cover_count[balls[c_new]] += 1
                        
                        new_critical = get_critical_count(cover_count)
                        delta_critical = new_critical - current_critical
                        
                        # Acceptance
                        if delta_critical < 0:
                            accept = True
                        else:
                            accept = random.random() < np.exp(-delta_critical / T)
                            
                        if accept:
                            # Commit
                            selected.remove(c_old)
                            selected_set.discard(c_old)
                            selected.append(c_new)
                            selected_set.add(c_new)
                            current_critical = new_critical
                            move_made = True
                            stall_count = 0
                        else:
                            # Revert
                            cover_count[balls[c_new]] -= 1
                            cover_count[ball_c_old] = old_counts_c_old
                            stall_count += 1
                            if stall_count > 50:
                                break

            if not move_made:
                # Stall handling
                if stall_count > 100:
                    break
            else:
                # Check for redundant removals after a swap
                # This helps to cascade improvements
                for c in list(selected):
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        selected.remove(c)
                        selected_set.discard(c)
                        current_critical = get_critical_count(cover_count)
                        move_made = True
                        
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected.copy()
                
            T *= alpha
            if T < T_min:
                break

    # Convert to list of digit lists
    if best_set is None:
        best_set = list(range(total))
        
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
