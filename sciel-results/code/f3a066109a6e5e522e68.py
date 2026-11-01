import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
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
    
    # Precompute balls using vectorized approach
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Generate all words as indices 0..6560
    # For each word, compute its 2*n neighbors
    words = np.arange(total, dtype=np.int32)
    
    # Compute digits for all words at once
    # digits[w, pos] = (w // weights[pos]) % 3
    digits = (words[:, None] // weights[None, :]) % 3  # shape (6561, 8)
    
    # Build balls array
    balls = np.zeros((total, ball_size), dtype=np.int32)
    balls[:, 0] = words
    
    idx = 1
    for pos in range(n):
        weight = weights[pos]
        digit = digits[:, pos]
        for new_digit in range(q):
            if new_digit == 0:
                mask = digit != 0
                diff = (-digit) * weight
            elif new_digit == 1:
                mask = digit != 1
                diff = (1 - digit) * weight
            else:  # new_digit == 2
                mask = digit != 2
                diff = (2 - digit) * weight
            
            # Only add where mask is true
            # We need to add this to balls[:, idx] for positions where mask is true
            # But we're filling sequentially... let's do it differently.
            # Actually, let's just build it properly.
            pass
    
    # Simpler: build balls in a loop but optimized
    balls = np.zeros((total, ball_size), dtype=np.int32)
    balls[:, 0] = words
    idx = 1
    for pos in range(n):
        weight = weights[pos]
        digit = digits[:, pos]
        for new_digit in range(q):
            if new_digit == 0:
                mask = digit != 0
                delta = (-digit) * weight
            elif new_digit == 1:
                mask = digit != 1
                delta = (1 - digit) * weight
            else:
                mask = digit != 2
                delta = (2 - digit) * weight
            
            # For all words where mask is true, add the neighbor
            # But we're filling a fixed column... this approach doesn't work directly
            # Let's use a different strategy: for each word, compute all neighbors
    
    # Most efficient: just loop over words but use numpy for digit extraction
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        temp = w
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit != digit:
                    balls[w, idx] = w + (new_digit - digit) * weight
                    idx += 1
    
    # Build inverse mapping: for each word, which codewords cover it
    # inv_lists[w] = list of codewords c such that w is in ball[c]
    inv_lists = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv_lists[w].append(c)
    
    # Convert to numpy arrays where possible
    # For the SA, we need fast access. Let's keep inv_lists as lists of arrays
    inv_arrays = [np.array(lst, dtype=np.int32) for lst in inv_lists]
    
    np.random.seed(42)
    random.seed(42)
    
    best_size = total
    best_set = None
    
    def greedy_init():
        selected = []
        selected_set = set()
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total
        cand_count = np.full(total, ball_size, dtype=np.int32)
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0 and time.time() < deadline - 15.0:
            # Find best candidate vectorized
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
            
            # Update cover counts and candidate counts
            ball = balls[best_idx]
            newly_covered = ball[cover_count[ball] == 0]
            if len(newly_covered) > 0:
                uncovered_count -= len(newly_covered)
                for w in newly_covered:
                    for c in inv_arrays[w]:
                        if active[c]:
                            cand_count[c] -= 1
            cover_count[ball] += 1
        
        return selected, selected_set, cover_count, uncovered_count
    
    def prune(selected, selected_set, cover_count):
        """Remove redundant codewords"""
        changed = True
        while changed and time.time() < deadline - 10.0:
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
        return selected, selected_set, cover_count
    
    def get_critical_count(cover_count):
        return np.sum(cover_count == 1)
    
    # Main search loop with multiple restarts
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        selected, selected_set, cover_count, uncovered_count = greedy_init()
        
        if uncovered_count > 0:
            continue
        
        # Prune
        selected, selected_set, cover_count = prune(selected, selected_set, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Simulated Annealing with swap moves
        T = 5.0
        T_min = 0.05
        alpha = 0.9998
        
        current_critical = get_critical_count(cover_count)
        stall_count = 0
        
        while time.time() < deadline - 3.0 and T > T_min:
            if len(selected) == 0:
                break
            
            c_old = selected[random.randint(0, len(selected) - 1)]
            ball_c_old = balls[c_old]
            
            # Find critical words in ball_c_old
            critical_mask = cover_count[ball_c_old] == 1
            critical_words = ball_c_old[critical_mask]
            
            if len(critical_words) == 0:
                # Can remove c_old
                cover_count[ball_c_old] -= 1
                selected.remove(c_old)
                selected_set.discard(c_old)
                current_critical = get_critical_count(cover_count)
                stall_count = 0
            else:
                # Need to find a swap partner
                # c_new must cover all critical_words
                # and must not be in selected_set
                
                if len(critical_words) == 1:
                    w = critical_words[0]
                    candidates = inv_arrays[w]
                    candidates = candidates[(candidates != c_old) & ~np.isin(candidates, list(selected_set))]
                elif len(critical_words) == 2:
                    w1, w2 = critical_words
                    set1 = set(inv_arrays[w1].tolist())
                    set2 = set(inv_arrays[w2].tolist())
                    common = set1 & set2
                    common.discard(c_old)
                    common -= selected_set
                    candidates = np.array(list(common), dtype=np.int32)
                elif len(critical_words) <= 5:
                    common = set(inv_arrays[critical_words[0]].tolist())
                    for w in critical_words[1:]:
                        common &= set(inv_arrays[w].tolist())
                        if not common:
                            break
                    common.discard(c_old)
                    common -= selected_set
                    candidates = np.array(list(common), dtype=np.int32)
                else:
                    candidates = np.array([], dtype=np.int32)
                
                if len(candidates) > 0:
                    c_new = candidates[random.randint(0, len(candidates) - 1)]
                    ball_c_new = balls[c_new]
                    
                    # Check validity: after removing c_old, no word should have count 0
                    # Words in ball_c_old but not in ball_c_new: count decreases by 1
                    # Words in ball_c_old and ball_c_new: count unchanged
                    # Words in ball_c_new but not in ball_c_old: count increases by 1
                    
                    # Simulate
                    old_counts_old = cover_count[ball_c_old].copy()
                    old_counts_new = cover_count[ball_c_new].copy()
                    
                    cover_count[ball_c_old] -= 1
                    if np.min(cover_count[ball_c_old]) < 0:
                        cover_count[ball_c_old] = old_counts_old
                        continue
                    
                    cover_count[ball_c_new] += 1
                    new_critical = get_critical_count(cover_count)
                    delta_critical = new_critical - current_critical
                    
                    if delta_critical < 0 or random.random() < np.exp(-delta_critical / T):
                        # Accept
                        selected.remove(c_old)
                        selected_set.discard(c_old)
                        selected.append(c_new)
                        selected_set.add(c_new)
                        current_critical = new_critical
                        stall_count = 0
                    else:
                        # Revert
                        cover_count[ball_c_new] -= 1
                        cover_count[ball_c_old] = old_counts_old
                    
                    # Try to remove redundant codewords
                    for c in list(selected):
                        if np.min(cover_count[balls[c]]) > 1:
                            cover_count[balls[c]] -= 1
                            selected.remove(c)
                            selected_set.discard(c)
                            current_critical = get_critical_count(cover_count)
                    
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_set = selected.copy()
                else:
                    stall_count += 1
                    if stall_count > 200:
                        break
            
            T *= alpha
    
    # Final polish: try to find more swaps with a lower temperature
    if best_set is not None and time.time() < deadline - 2.0:
        # Reconstruct cover counts from best_set
        cover_count = np.zeros(total, dtype=np.int32)
        for c in best_set:
            cover_count[balls[c]] += 1
        
        selected = list(best_set)
        selected_set = set(selected)
        current_critical = get_critical_count(cover_count)
        
        T = 0.1
        T_min = 0.01
        alpha = 0.999
        
        stall_count = 0
        while time.time() < deadline - 1.0 and T > T_min:
            if len(selected) == 0:
                break
            
            c_old = selected[random.randint(0, len(selected) - 1)]
            ball_c_old = balls[c_old]
            critical_mask = cover_count[ball_c_old] == 1
            critical_words = ball_c_old[critical_mask]
            
            if len(critical_words) == 0:
                cover_count[ball_c_old] -= 1
                selected.remove(c_old)
                selected_set.discard(c_old)
                current_critical = get_critical_count(cover_count)
                stall_count = 0
            else:
                if len(critical_words) == 1:
                    w = critical_words[0]
                    candidates = inv_arrays[w]
                    candidates = candidates[(candidates != c_old) & ~np.isin(candidates, list(selected_set))]
                elif len(critical_words) == 2:
                    w1, w2 = critical_words
                    set1 = set(inv_arrays[w1].tolist())
                    set2 = set(inv_arrays[w2].tolist())
                    common = set1 & set2
                    common.discard(c_old)
                    common -= selected_set
                    candidates = np.array(list(common), dtype=np.int32)
                elif len(critical_words) <= 5:
                    common = set(inv_arrays[critical_words[0]].tolist())
                    for w in critical_words[1:]:
                        common &= set(inv_arrays[w].tolist())
                        if not common:
                            break
                    common.discard(c_old)
                    common -= selected_set
                    candidates = np.array(list(common), dtype=np.int32)
                else:
                    candidates = np.array([], dtype=np.int32)
                
                if len(candidates) > 0:
                    c_new = candidates[random.randint(0, len(candidates) - 1)]
                    ball_c_new = balls[c_new]
                    
                    old_counts_old = cover_count[ball_c_old].copy()
                    old_counts_new = cover_count[ball_c_new].copy()
                    
                    cover_count[ball_c_old] -= 1
                    if np.min(cover_count[ball_c_old]) < 0:
                        cover_count[ball_c_old] = old_counts_old
                        continue
                    
                    cover_count[ball_c_new] += 1
                    new_critical = get_critical_count(cover_count)
                    delta_critical = new_critical - current_critical
                    
                    if delta_critical < 0 or random.random() < np.exp(-delta_critical / T):
                        selected.remove(c_old)
                        selected_set.discard(c_old)
                        selected.append(c_new)
                        selected_set.add(c_new)
                        current_critical = new_critical
                        stall_count = 0
                        
                        for c in list(selected):
                            if np.min(cover_count[balls[c]]) > 1:
                                cover_count[balls[c]] -= 1
                                selected.remove(c)
                                selected_set.discard(c)
                                current_critical = get_critical_count(cover_count)
                        
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_set = selected.copy()
                    else:
                        cover_count[ball_c_new] -= 1
                        cover_count[ball_c_old] = old_counts_old
                else:
                    stall_count += 1
                    if stall_count > 100:
                        break
            
            T *= alpha
    
    if best_set is None:
        best_set = list(range(total))
    
    # Convert to list of digit lists
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
