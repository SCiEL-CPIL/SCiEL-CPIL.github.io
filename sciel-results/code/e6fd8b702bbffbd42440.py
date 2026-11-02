import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls as a numpy array
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls_arr = np.zeros((N, ball_size), dtype=np.int32)
    for w in range(N):
        balls_arr[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls_arr[w, idx] = new_w
                idx += 1
    
    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]
    
    def prune(selected, cover_count):
        """Remove codewords that are not essential."""
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(selected):
                c = selected[i]
                ball = balls_arr[c]
                if np.min(cover_count[ball]) > 1:
                    cover_count[ball] -= 1
                    selected.pop(i)
                    changed = True
                else:
                    i += 1
        return selected
    
    def dual_greedy_construct(cover_count, uncovered):
        """
        Dual greedy set cover: prioritize words with highest 'shadow prices'
        (inverse frequency of remaining candidate coverers).
        
        At each step:
        1. For each uncovered word, count how many candidate codewords can cover it
           (i.e., how many codewords' balls contain it and haven't been selected yet)
        2. Compute a 'scarcity score' for each uncovered word = 1 / (number of candidates)
        3. For each candidate codeword, compute weighted gain = sum of scarcity scores
           of uncovered words it covers
        4. Pick the codeword with highest weighted gain
        """
        selected = []
        # Track which codewords have been selected (for candidate counting)
        selected_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 10.0:
                break
            
            # Get indices of uncovered words
            uncovered_idx = np.where(uncovered)[0]
            if len(uncovered_idx) == 0:
                break
            
            # For each uncovered word, count how many candidate codewords can cover it
            # A codeword c can cover word w if w is in balls_arr[c]
            # Equivalently, c is in balls_arr[w]
            # We need to count, for each uncovered word, how many of its 17 neighbors
            # are NOT yet selected
            
            # For efficiency, precompute: for each word, the list of candidate codewords
            # is balls_arr[word] (the 17 words within distance 1)
            # Count how many of those are not in selected_set
            
            # Vectorized approach: for each uncovered word, count non-selected candidates
            # balls_arr[uncovered_idx] gives shape (len(uncovered_idx), 17)
            # We need to check which of these are NOT in selected_set
            
            # Build a boolean mask of selected codewords
            selected_mask = np.zeros(N, dtype=bool)
            for c in selected_set:
                selected_mask[c] = True
            
            # For each uncovered word, its candidates are balls_arr[word]
            # Count how many candidates are not selected
            candidate_balls = balls_arr[uncovered_idx]  # shape (num_uncovered, 17)
            # Check which candidates are NOT selected
            not_selected = ~selected_mask[candidate_balls]  # shape (num_uncovered, 17)
            candidate_counts = not_selected.sum(axis=1)  # shape (num_uncovered,)
            
            # Avoid division by zero: if a word has 0 candidates, it's uncovered forever
            # (shouldn't happen in practice, but guard against it)
            candidate_counts = np.maximum(candidate_counts, 1)
            
            # Scarcity score = 1 / candidate_count (higher = more scarce)
            scarcity = 1.0 / candidate_counts.astype(np.float64)
            
            # For each candidate codeword, compute weighted gain
            # A codeword c covers word w if w is in balls_arr[c]
            # We want: for each codeword c, sum of scarcity[w] for all uncovered w in balls_arr[c]
            
            # Build a scarcity array indexed by word
            scarcity_full = np.zeros(N, dtype=np.float64)
            scarcity_full[uncovered_idx] = scarcity
            
            # For each codeword, weighted gain = sum of scarcity_full over its ball
            # balls_arr has shape (N, 17), so we can do:
            gains = scarcity_full[balls_arr].sum(axis=1)  # shape (N,)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Among codewords with max gain, pick one that also covers some scarce words
            # Use a secondary criterion: prefer codewords that cover words with higher scarcity
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                # Secondary: prefer codewords whose ball contains words with highest min scarcity
                # For each candidate, compute the minimum scarcity in its ball
                # This is a tie-breaker to prefer more "central" codewords
                min_scarcity_in_ball = scarcity_full[balls_arr[candidates]].min(axis=1)
                # Pick the one with highest min scarcity (most centrally located among scarce words)
                best_local = np.argmax(min_scarcity_in_ball)
                best_idx = candidates[best_local]
            else:
                best_idx = candidates[0]
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            ball = balls_arr[best_idx]
            cover_count[ball] += 1
            uncovered[ball] = False
        
        return selected
    
    def local_search_swap(selected, cover_count, deadline):
        """Try random swaps to enable further pruning."""
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            if not selected:
                break
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                i = np.random.randint(len(selected))
                u = selected[i]
                v = np.random.randint(N)
                if v == u:
                    continue
                ball_u = balls_arr[u]
                ball_v = balls_arr[v]
                # Check safety
                ball_u_set = set(ball_u.tolist())
                ball_v_set = set(ball_v.tolist())
                risk = ball_u_set - ball_v_set
                if risk:
                    risk_arr = np.array(list(risk), dtype=np.int32)
                    if np.min(cover_count[risk_arr]) < 2:
                        continue
                # Safe swap
                cover_count[ball_u] -= 1
                cover_count[ball_v] += 1
                selected[i] = v
                new_len_before = len(selected)
                selected = prune(selected, cover_count)
                if len(selected) < new_len_before:
                    improved = True
                    break
                else:
                    cover_count[ball_v] -= 1
                    cover_count[ball_u] += 1
                    selected[i] = u
        return selected
    
    def local_search_2for1(selected, cover_count, deadline):
        """Try to replace two codewords with one."""
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            K = len(selected)
            if K < 2:
                break
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                i1 = np.random.randint(K)
                i2 = np.random.randint(K)
                if i1 == i2:
                    continue
                if i1 > i2:
                    i1, i2 = i2, i1
                c1 = selected[i1]
                c2 = selected[i2]
                ball1 = balls_arr[c1]
                ball2 = balls_arr[c2]
                
                # Find words that would become uncovered if we remove both c1 and c2
                # and check if a single codeword can cover them
                cc_test = cover_count.copy()
                cc_test[ball1] -= 1
                cc_test[ball2] -= 1
                newly_uncovered = np.where(cc_test == 0)[0]
                
                if len(newly_uncovered) == 0:
                    # Both can be removed entirely
                    cover_count[ball1] -= 1
                    cover_count[ball2] -= 1
                    selected.pop(max(i1, i2))
                    selected.pop(min(i1, i2))
                    selected = prune(selected, cover_count)
                    improved = True
                    break
                else:
                    # Find a codeword that covers all newly_uncovered words
                    # The candidate must be in the intersection of balls of all newly_uncovered
                    # For small sets, compute intersection
                    if len(newly_uncovered) <= 6:
                        candidates = set(balls_arr[newly_uncovered[0]].tolist())
                        for uw in newly_uncovered[1:]:
                            candidates.intersection_update(balls_arr[uw].tolist())
                            if not candidates:
                                break
                        candidates.discard(c1)
                        candidates.discard(c2)
                        if candidates:
                            v = random.choice(list(candidates))
                            ball_v = balls_arr[v]
                            # Verify safety
                            combined_risk = set(ball1.tolist()) | set(ball2.tolist())
                            combined_risk.difference_update(set(ball_v.tolist()))
                            if combined_risk:
                                risk_arr = np.array(list(combined_risk), dtype=np.int32)
                                if np.min(cc_test[risk_arr]) < 1:
                                    continue
                            # Perform 2-for-1
                            cover_count[ball1] -= 1
                            cover_count[ball2] -= 1
                            cover_count[ball_v] += 1
                            selected.pop(max(i1, i2))
                            selected.pop(min(i1, i2))
                            selected.append(v)
                            selected = prune(selected, cover_count)
                            improved = True
                            break
    
    best_size = N
    best_set = None
    
    # Main search loop with multiple restarts
    while time.time() < deadline - 10.0:
        # 1. Dual Greedy Construction
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered = np.ones(N, dtype=bool)
        selected = dual_greedy_construct(cover_count, uncovered)
        
        if np.any(uncovered):
            continue
        
        # 2. Initial Pruning
        selected = prune(selected, cover_count)
        
        # 3. Local Search: Swaps
        selected = local_search_swap(selected, cover_count, deadline)
        
        # 4. Local Search: 2-for-1
        selected = local_search_2for1(selected, cover_count, deadline)
        
        # 5. Final Pruning
        selected = prune(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # If close to deadline, do final polish
        if time.time() > deadline - 20.0 and best_set is not None:
            cover_count_best = np.zeros(N, dtype=np.int32)
            for c in best_set:
                cover_count_best[balls_arr[c]] += 1
            best_set = prune(best_set, cover_count_best)
            best_set = local_search_swap(best_set, cover_count_best, deadline)
            best_set = local_search_2for1(best_set, cover_count_best, deadline)
            best_set = prune(best_set, cover_count_best)
            if len(best_set) < best_size:
                best_size = len(best_set)
                best_set = best_set.copy()
            break
    
    if best_set is None:
        return [idx_to_list(w) for w in range(N)]
        
    return [idx_to_list(w) for w in best_set]
