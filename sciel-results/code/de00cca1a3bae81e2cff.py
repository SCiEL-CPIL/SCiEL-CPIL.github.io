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
    
    # Precompute balls using positional weights
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
    
    # Build inverse mapping: for each word, which codewords cover it
    inv_lists = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv_lists[w].append(c)
    
    # Convert inv_lists to numpy arrays for faster access
    inv_arrays = [np.array(lst, dtype=np.int32) for lst in inv_lists]
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    def greedy_init():
        """Fast greedy initialization using vectorized gain computation."""
        selected = []
        selected_set = set()
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total
        
        # Track which codewords are still active (not yet selected)
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0 and time.time() < deadline - 12.0:
            # Compute gain for each active codeword: number of uncovered words it covers
            # Vectorized: for each candidate c, count uncovered words in ball[c]
            gains = np.zeros(total, dtype=np.int32)
            for c in range(total):
                if not active[c]:
                    continue
                gains[c] = np.sum(cover_count[balls[c]] == 0)
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            
            candidates = np.where((gains == max_gain) & active)[0]
            if len(candidates) == 0:
                break
            
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            active[best_idx] = False
            
            # Update coverage
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                cover_count[w] += 1
                
        return selected, selected_set, cover_count, uncovered_count
    
    def prune_redundant(selected, selected_set, cover_count):
        """Remove all redundant codewords (those whose ball is fully covered by others)."""
        changed = True
        while changed and time.time() < deadline - 6.0:
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
    
    def local_search(selected, selected_set, cover_count):
        """Swap-based local search with periodic pruning."""
        T = 5.0
        T_min = 0.05
        alpha = 0.99995
        stall_count = 0
        
        def get_critical_count(cover_count):
            return int(np.sum(cover_count == 1))
        
        current_critical = get_critical_count(cover_count)
        
        while time.time() < deadline - 3.0 and T > T_min:
            if len(selected) == 0:
                break
            
            c_old = selected[random.randint(0, len(selected) - 1)]
            ball_c_old = balls[c_old]
            critical_words = ball_c_old[cover_count[ball_c_old] == 1]
            
            if len(critical_words) == 0:
                # Can remove directly
                cover_count[ball_c_old] -= 1
                selected.remove(c_old)
                selected_set.discard(c_old)
                current_critical = get_critical_count(cover_count)
                stall_count = 0
            else:
                # Need to find a replacement
                candidates = None
                if len(critical_words) == 1:
                    w = critical_words[0]
                    candidates = [c for c in inv_arrays[w] if c != c_old and c not in selected_set]
                elif len(critical_words) == 2:
                    w1, w2 = critical_words[0], critical_words[1]
                    set1 = set(inv_arrays[w1].tolist())
                    set2 = set(inv_arrays[w2].tolist())
                    candidates = [c for c in set1.intersection(set2) if c != c_old and c not in selected_set]
                elif len(critical_words) <= 4:
                    common = None
                    for w in critical_words:
                        if common is None:
                            common = set(inv_arrays[w].tolist())
                        else:
                            common.intersection_update(inv_arrays[w].tolist())
                    if common:
                        candidates = [c for c in common if c != c_old and c not in selected_set]
                    else:
                        candidates = []
                else:
                    candidates = []
                
                if not candidates:
                    stall_count += 1
                    if stall_count > 200:
                        break
                    T *= alpha
                    continue
                
                c_new = candidates[random.randint(0, len(candidates) - 1)]
                
                # Simulate the swap
                old_counts_c_old = cover_count[ball_c_old].copy()
                old_counts_c_new = cover_count[balls[c_new]].copy()
                
                cover_count[ball_c_old] -= 1
                if np.min(cover_count[ball_c_old]) < 0:
                    cover_count[ball_c_old] = old_counts_c_old
                    stall_count += 1
                    if stall_count > 200:
                        break
                    T *= alpha
                    continue
                cover_count[balls[c_new]] += 1
                
                new_critical = get_critical_count(cover_count)
                delta_critical = new_critical - current_critical
                
                if delta_critical < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta_critical / T)
                
                if accept:
                    selected.remove(c_old)
                    selected_set.discard(c_old)
                    selected.append(c_new)
                    selected_set.add(c_new)
                    current_critical = new_critical
                    stall_count = 0
                else:
                    cover_count[balls[c_new]] -= 1
                    cover_count[ball_c_old] = old_counts_c_old
                    stall_count += 1
                    if stall_count > 200:
                        break
                
                # Periodic pruning after every 10 accepted moves
                if stall_count == 0 and len(selected) % 10 == 0:
                    selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
                    current_critical = get_critical_count(cover_count)
            
            T *= alpha
            if T < T_min:
                break
        
        return selected, selected_set, cover_count
    
    # Main search loop: multiple restarts with greedy init + local search
    while time.time() < deadline - 5.0:
        selected, selected_set, cover_count, uncovered_count = greedy_init()
        
        if uncovered_count > 0:
            continue
        
        # Prune initial solution
        selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Local search
        selected, selected_set, cover_count = local_search(selected, selected_set, cover_count)
        
        # Final prune after local search
        selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # If we're doing well, do another round of local search from current state
        if len(selected) <= best_size + 2 and time.time() < deadline - 10.0:
            selected, selected_set, cover_count = local_search(selected, selected_set, cover_count)
            selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected.copy()
    
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
