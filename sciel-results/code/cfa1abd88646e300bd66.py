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
    
    # Build inverse mapping
    inv_lists = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv_lists[w].append(c)
    
    # Convert to numpy arrays for faster access if needed, but lists are fine for small sizes
    # We will use a set for selected codewords for O(1) membership
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    # Initial Greedy Construction
    # We do a fast greedy to get a valid starting point
    def greedy_init():
        selected = []
        selected_set = set()
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total
        
        # cand_count[c] = number of uncovered words in ball[c]
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            cand_count[c] = ball_size
            
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0 and time.time() < deadline - 10.0:
            # Find best candidate
            # Vectorized max
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)
            if max_count <= 0:
                break
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            # Random pick among best
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            active[best_idx] = False
            
            # Update
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1
                
        return selected, selected_set, cover_count, uncovered_count

    # Main SA Loop
    while time.time() < deadline - 5.0:
        selected, selected_set, cover_count, uncovered_count = greedy_init()
        
        if uncovered_count > 0:
            continue
            
        # Prune initial solution
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
        T = 10.0
        T_min = 0.1
        alpha = 0.9999
        
        # Calculate initial critical count
        def get_critical_count(cover_count):
            return np.sum(cover_count == 1)
            
        current_critical = get_critical_count(cover_count)
        
        stall_count = 0
        
        while time.time() < deadline - 2.0 and T > T_min:
            move_made = False
            # Try a random move
            if len(selected) == 0:
                break
                
            c_old = selected[random.randint(0, len(selected) - 1)]
            
            # Determine words that would become uncovered if c_old is removed
            # These are words in ball[c_old] with cover_count == 1
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
                # And c_new must not be in selected_set
                # And c_new != c_old
                
                # Find candidates
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
                    # Small intersection
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
                    c_new = candidates[random.randint(0, len(candidates) - 1)]
                    
                    # Check validity of swap:
                    # Remove c_old, Add c_new
                    # Words in ball[c_old] \ ball[c_new] must have cover_count > 1 (after removing c_old)
                    # Words in ball[c_new] \ ball[c_old] can have cover_count >= 0
                    # Words in intersection have cover_count >= 1 (since c_old covered them, and we add c_new, so they are fine)
                    
                    # Actually, simpler:
                    # 1. Decrement cover_count for ball[c_old]
                    # 2. Check if any word has cover_count == 0
                    # 3. If yes, invalid.
                    # 4. If no, increment cover_count for ball[c_new]
                    # 5. Calculate new critical count
                    
                    # To avoid modifying and reverting often, we can simulate:
                    # New cover count for w:
                    # if w in ball[c_old] and w in ball[c_new]: cover_count[w] (unchanged)
                    # if w in ball[c_old] and w not in ball[c_new]: cover_count[w] - 1
                    # if w not in ball[c_old] and w in ball[c_new]: cover_count[w] + 1
                    # if w not in ball[c_old] and w not in ball[c_new]: cover_count[w] (unchanged)
                    
                    # We only care about:
                    # 1. Validity: min(cover_count[w] - 1) for w in ball[c_old] \ ball[c_new] must be >= 1
                    #    i.e., min(cover_count[w]) for w in ball[c_old] \ ball[c_new] must be >= 2
                    # 2. New Critical Count:
                    #    - Old critical words in ball[c_old] \ ball[c_new] become non-critical (count 2 -> 1? No, they were 1, become 0 -> Invalid)
                    #      Wait, if they were critical (count 1) and in ball[c_old] \ ball[c_new], they become 0. Invalid.
                    #      So validity check covers this.
                    #    - Old critical words in ball[c_old] \cap ball[c_new] remain critical (count 1).
                    #    - Old non-critical words in ball[c_old] \ ball[c_new]:
                    #      If count was 2, becomes 1 (new critical).
                    #      If count > 2, stays non-critical.
                    #    - Old non-critical words in ball[c_new] \ ball[c_old]:
                    #      If count was 0, becomes 1 (new critical).
                    #      If count >= 1, stays non-critical (or count increases).
                    
                    # Let's just do the update and revert if needed, it's fast enough for 17 elements.
                    
                    # Store old values
                    old_counts_c_old = cover_count[ball_c_old].copy()
                    old_counts_c_new = cover_count[balls[c_new]].copy()
                    
                    # Apply
                    cover_count[ball_c_old] -= 1
                    if np.min(cover_count[ball_c_old]) < 0:
                        # Invalid, revert
                        cover_count[ball_c_old] = old_counts_c_old
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
                        
            if not move_made:
                stall_count += 1
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
                        move_made = True # Reset stall?
                        
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
