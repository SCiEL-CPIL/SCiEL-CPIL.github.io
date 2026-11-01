import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (invalid for score but valid format)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls: balls[w, 0] = w, balls[w, 1:17] = neighbors
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
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
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Precompute word priority based on digit diversity
    word_priority = np.zeros(total, dtype=np.float64)
    for w in range(total):
        temp = w
        digits = []
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        distinct = len(set(digits))
        spread = max(digits) - min(digits) if digits else 0
        word_priority[w] = distinct * 1000 + spread * 100 + (total - w)
    
    # Step 1: Greedy Construction
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    rng = np.random.default_rng(42)
    
    while np.any(current_uncovered) and time.time() < deadline - 30.0:
        counts = np.sum(current_uncovered[balls], axis=1)
        valid_mask = counts > 0
        if not np.any(valid_mask):
            break
        
        max_count = np.max(counts[valid_mask])
        candidates = np.where((counts == max_count) & valid_mask)[0]
        
        if len(candidates) == 1:
            best_idx = candidates[0]
        else:
            prios = word_priority[candidates]
            weights_sel = prios / (prios.sum() + 1e-9)
            best_idx = candidates[rng.choice(len(candidates), p=weights_sel)]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
    
    selected_list = list(selected)
    
    # Step 2: Aggressive Pruning - remove any codeword whose ball is fully covered
    changed = True
    while changed and time.time() < deadline - 25.0:
        changed = False
        i = 0
        while i < len(selected_list):
            c = selected_list[i]
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                selected_list.pop(i)
                changed = True
            else:
                i += 1
    
    # Step 3: Simulated Annealing
    # Objective: minimize |selected| + large_penalty * |uncovered|
    # Moves: remove a codeword, add a codeword, or swap
    selected_set = set(selected_list)
    best_size = len(selected_list)
    best_solution = list(selected_list)
    best_cover = cover_count.copy()
    
    T_start = 50.0  # Initial temperature
    T_end = 0.1     # Final temperature
    cooling_rate = (T_end / T_start) ** (1.0 / 2000)
    T = T_start
    
    penalty = 1000  # Penalty per uncovered word
    
    # Track current state
    cur_cover = cover_count.copy()
    cur_selected = list(selected_list)
    cur_uncovered_count = int(np.sum(cur_cover == 0))
    cur_cost = len(cur_selected) + penalty * cur_uncovered_count
    
    best_cost = cur_cost
    
    max_iter = 5000
    iter_count = 0
    
    while time.time() < deadline - 5.0 and iter_count < max_iter:
        iter_count += 1
        T *= cooling_rate
        
        if T < T_end:
            T = T_end
        
        # Choose a random move type
        move_type = rng.random()
        
        if move_type < 0.3 and len(cur_selected) > 0:
            # Remove a random codeword
            idx = rng.integers(0, len(cur_selected))
            c = cur_selected[idx]
            cur_cover[balls[c]] -= 1
            new_uncovered = int(np.sum(cur_cover == 0))
            new_cost = (len(cur_selected) - 1) + penalty * new_uncovered
            delta_cost = new_cost - cur_cost
            
            if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / max(T, 1e-9)):
                cur_selected.pop(idx)
                cur_uncovered_count = new_uncovered
                cur_cost = new_cost
                
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_size = len(cur_selected)
                    best_solution = list(cur_selected)
                    best_cover = cur_cover.copy()
            else:
                cur_cover[balls[c]] += 1
                
        elif move_type < 0.6:
            # Add a random codeword not already selected
            # Sample from words with low coverage to find useful additions
            # For efficiency, sample a few candidates
            found_added = False
            for attempt in range(20):
                c = rng.integers(0, total)
                if c in selected_set:
                    continue
                # Check if this helps
                new_cover = cur_cover.copy()
                new_cover[balls[c]] += 1
                new_uncovered = int(np.sum(new_cover == 0))
                new_cost = (len(cur_selected) + 1) + penalty * new_uncovered
                delta_cost = new_cost - cur_cost
                
                if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / max(T, 1e-9)):
                    cur_selected.append(c)
                    selected_set.add(c)
                    cur_cover = new_cover
                    cur_uncovered_count = new_uncovered
                    cur_cost = new_cost
                    found_added = True
                    
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_size = len(cur_selected)
                        best_solution = list(cur_selected)
                        best_cover = cur_cover.copy()
                    break
            
        else:
            # Swap: remove one, add another
            if len(cur_selected) > 0:
                idx = rng.integers(0, len(cur_selected))
                c_remove = cur_selected[idx]
                
                # Try to find a good replacement
                # Words that cover the uniquely covered words by c_remove
                unique_to_c = balls[c_remove]
                unique_mask = cur_cover[unique_to_c] == 1
                unique_words = unique_to_c[unique_mask]
                
                if len(unique_words) == 0:
                    # Pure removal candidate
                    cur_cover[balls[c_remove]] -= 1
                    new_uncovered = int(np.sum(cur_cover == 0))
                    new_cost = (len(cur_selected) - 1) + penalty * new_uncovered
                    delta_cost = new_cost - cur_cost
                    
                    if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / max(T, 1e-9)):
                        cur_selected.pop(idx)
                        selected_set.discard(c_remove)
                        cur_uncovered_count = new_uncovered
                        cur_cost = new_cost
                        
                        if cur_cost < best_cost:
                            best_cost = cur_cost
                            best_size = len(cur_selected)
                            best_solution = list(cur_selected)
                            best_cover = cur_cover.copy()
                    else:
                        cur_cover[balls[c_remove]] += 1
                else:
                    # Find candidates that cover all unique words
                    # Intersect the balls of unique words
                    candidate_set = set(balls[unique_words[0]].tolist())
                    for w in unique_words[1:]:
                        candidate_set &= set(balls[w].tolist())
                    
                    if candidate_set:
                        candidates = [c for c in candidate_set if c not in selected_set]
                        if candidates:
                            c_add = candidates[rng.integers(0, len(candidates))]
                            cur_cover[balls[c_remove]] -= 1
                            cur_cover[balls[c_add]] += 1
                            new_uncovered = int(np.sum(cur_cover == 0))
                            new_cost = len(cur_selected) + penalty * new_uncovered
                            delta_cost = new_cost - cur_cost
                            
                            if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / max(T, 1e-9)):
                                cur_selected[idx] = c_add
                                selected_set.discard(c_remove)
                                selected_set.add(c_add)
                                cur_uncovered_count = new_uncovered
                                cur_cost = new_cost
                                
                                if cur_cost < best_cost:
                                    best_cost = cur_cost
                                    best_size = len(cur_selected)
                                    best_solution = list(cur_selected)
                                    best_cover = cur_cover.copy()
                            else:
                                cur_cover[balls[c_remove]] += 1
                                cur_cover[balls[c_add]] -= 1
    
    # Use the best solution found
    if best_uncovered_count := int(np.sum(best_cover == 0)) == 0:
        final_selected = best_solution
        final_cover = best_cover
    else:
        # Fall back to current if it's valid, otherwise use selected_list
        if int(np.sum(cur_cover == 0)) == 0:
            final_selected = cur_selected
            final_cover = cur_cover
        else:
            final_selected = selected_list
            final_cover = cover_count
    
    # Final pruning pass
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        i = 0
        while i < len(final_selected):
            c = final_selected[i]
            if np.min(final_cover[balls[c]]) > 1:
                final_cover[balls[c]] -= 1
                final_selected.pop(i)
                changed = True
            else:
                i += 1
    
    # Verify coverage
    if int(np.sum(final_cover == 0)) > 0:
        # Need to fix coverage - add words to cover uncovered
        uncovered = np.where(final_cover == 0)[0]
        for w in uncovered:
            # Find a neighbor not in final_selected
            for nb in balls[w]:
                if nb not in set(final_selected):
                    final_selected.append(nb)
                    final_cover[balls[nb]] += 1
                    break
    
    # Convert to list of digit lists
    result = []
    for w in final_selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
