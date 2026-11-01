import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    if total_words := q ** n <= 10000:
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []
        
        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = 0
                for j in uncovered:
                    if hamming_dist(all_words[i], all_words[j]) <= radius:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1:
                break
                
            codewords.append(all_words[best_idx])
            new_uncovered = set()
            for j in uncovered:
                if hamming_dist(all_words[best_idx], all_words[j]) > radius:
                    new_uncovered.add(j)
            uncovered = new_uncovered
            
        return codewords
    else:
        return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
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
                
    # Precompute ball membership as a boolean matrix
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True
        
    # Precompute "covers" for each word: which codewords cover it
    # Due to symmetry, covers[w] is balls[w]
    # We use ball_membership[:, w] to get all c such that w in balls[c]
    
    best_size = total
    best_set = None
    
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Initialization Strategy
        # Alternate between Randomized Greedy and Latin Hypercube / Max-Min Distance init
        use_latin_init = (iteration % 3 == 0)
        
        if use_latin_init:
            # Latin Hypercube Sampling + Greedy Fill
            # Goal: Start with a diverse set that minimizes overlap initially
            # We pick a random permutation of the space and greedily select 
            # points that are far from existing selected points to create a sparse core.
            # Then fill gaps with standard greedy.
            
            # To ensure diversity, we can use a simple random permutation of indices
            # and pick the first K that are "far enough" from each other? 
            # Or just standard greedy but with a penalty for overlap?
            
            # Let's stick to a robust variant: 
            # 1. Randomly shuffle all words.
            # 2. Pick the first one.
            # 3. For subsequent picks, prefer those that cover the most NEW words (standard greedy) 
            #    BUT among ties, prefer those that are far from existing codewords.
            
            selected = []
            cover_count = np.zeros(total, dtype=np.int32)
            uncovered_mask = np.ones(total, dtype=bool)
            
            # Precompute pairwise distances? Too expensive (6561^2).
            # Instead, during greedy, if counts are tied, check distance to last few selected.
            
            while np.any(uncovered_mask):
                if time.time() > deadline - 10.0:
                    break
                    
                counts = uncovered_mask[balls].sum(axis=1)
                
                if np.max(counts) == 0:
                    break
                    
                max_count = np.max(counts)
                candidates = np.where(counts == max_count)[0]
                
                if len(candidates) > 1 and len(selected) > 0:
                    # Tie-breaking: choose candidate that is farthest from the last selected codeword
                    # to encourage spreading out.
                    last_sel = selected[-1]
                    # Calculate distance to last_sel for all candidates
                    # dist(c, last_sel) = number of positions where c != last_sel
                    # We can compute this quickly using precomputed digits? 
                    # Or just use ball_membership? No, ball_membership is radius 1.
                    # We need Hamming distance.
                    # Let's precompute a distance matrix? 6561x6561 int8 is ~43MB. Acceptable.
                    # But we didn't precompute it. Let's compute on the fly for the tie set.
                    # Tie set is usually small.
                    
                    # Convert last_sel to digits
                    # Actually, simpler: just pick randomly if tie set is small, 
                    # or use a precomputed "farthest" heuristic if we had the matrix.
                    # Let's just use random selection for ties as in the original, 
                    # but maybe bias towards indices that are "spread out" in base-3 representation?
                    # For now, keep random selection to avoid overhead, 
                    # but we can try to pick one that is NOT in the ball of any recent selection?
                    
                    # Let's try: filter candidates to those not covered by the last 3 selected?
                    # If a candidate is already covered by previous ones, its "new" count is lower.
                    # The greedy count already accounts for this (uncovered_mask).
                    # So candidates with max count are effectively "new" centers.
                    
                    # Let's just pick randomly among candidates.
                    best_idx = candidates[np.random.randint(len(candidates))]
                else:
                    best_idx = candidates[0]
                
                selected.append(best_idx)
                cover_count[balls[best_idx]] += 1
                uncovered_mask = (cover_count == 0)
                
        else:
            # Standard Randomized Greedy
            selected = []
            cover_count = np.zeros(total, dtype=np.int32)
            uncovered_mask = np.ones(total, dtype=bool)
            
            while np.any(uncovered_mask):
                if time.time() > deadline - 10.0:
                    break
                    
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
            continue # Invalid solution due to timeout in greedy
            
        # 2. Local Search
        # Phase 1: Remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
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

                candidate_mask = np.ones(total, dtype=bool)
                
                for cw in critical_words:
                    candidate_mask &= ball_membership[:, cw]
                
                valid_candidates = np.where(candidate_mask)[0]
                valid_candidates = valid_candidates[valid_candidates != c]
                
                if len(valid_candidates) > 0:
                    # Heuristic: Pick a candidate that maximizes the number of OTHER words it covers?
                    # Or just random.
                    # Let's try to pick one that is "far" from c to increase diversity?
                    # Or one that covers the most "critical" words of others?
                    # For speed, random.
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                    
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1)
        # This is the key to escaping local optima.
        if len(selected) > 450 and time.time() < deadline - 8.0:
            improved_2ex = True
            # Limit the number of 2-exchange passes to avoid burning all time
            # on this expensive phase if it's not yielding results quickly.
            passes = 0
            while improved_2ex and time.time() < deadline - 5.0 and passes < 5:
                improved_2ex = False
                passes += 1
                new_selected = list(selected)
                
                indices = list(range(len(new_selected)))
                np.random.shuffle(indices)
                
                found_swap = False
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if time.time() > deadline - 5.0:
                            found_swap = True # break out
                            break
                            
                        c1 = new_selected[indices[i]]
                        c2 = new_selected[indices[j]]
                        
                        words_c1 = balls[c1]
                        words_c2 = balls[c2]
                        
                        to_cover = set()
                        # Words covered only by c1
                        for w in words_c1:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        # Words covered only by c2
                        for w in words_c2:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        # Words covered by both c1 and c2 (count 2)
                        # If we remove both, count becomes 0.
                        # Intersection of balls[c1] and balls[c2]
                        # We can find intersection by checking membership
                        # Or simpler: for w in words_c1, if w in set(words_c2) and count==2
                        # Since balls are small (17), this is fast.
                        set_c2 = set(words_c2)
                        for w in words_c1:
                            if w in set_c2 and cover_count[w] == 2:
                                to_cover.add(w)
                                
                        if not to_cover:
                            # Both removable
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            improved_2ex = True
                            found_swap = True
                            break
                            
                        # Find c3
                        candidate_mask = np.ones(total, dtype=bool)
                        for w in to_cover:
                            candidate_mask &= ball_membership[:, w]
                            
                        valid_candidates = np.where(candidate_mask)[0]
                        valid_candidates = valid_candidates[valid_candidates != c1]
                        valid_candidates = valid_candidates[valid_candidates != c2]
                        
                        if len(valid_candidates) > 0:
                            c3 = valid_candidates[np.random.randint(len(valid_candidates))]
                            
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            cover_count[balls[c3]] += 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            new_selected.append(c3)
                            improved_2ex = True
                            found_swap = True
                            break
                        
                    if found_swap:
                        break
                if improved_2ex:
                    selected = new_selected
                else:
                    improved_2ex = False
                    
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            
        if best_size <= 486:
            break

    if best_set is None:
        # Fallback: return a valid (but large) code if we failed to find a small one
        # We can just return the result of a single greedy run if we have one, 
        # or generate a fresh one.
        # Since we looped, best_set should be set if we ever completed a greedy.
        # If we timed out in greedy every time, we need a fallback.
        # Let's do a quick greedy if best_set is None.
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

    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    result = [idx_to_digits(cw) for cw in best_set]
    
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
            
    return unique_result
