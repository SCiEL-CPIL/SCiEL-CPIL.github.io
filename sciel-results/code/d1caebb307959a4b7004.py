import time
import numpy as np
import itertools
import random

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
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Weights for positions (big-endian)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
    # balls[w] is a list of 17 neighbors
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
    
    # Helper to convert index to list of digits
    def idx_to_digits(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return list(reversed(digits))
    
    # 1. Greedy Construction
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    while np.any(current_uncovered):
        if time.time() > deadline - 30.0:
            break
            
        # Count uncovered neighbors for each word
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        
        if max_count == 0:
            break
        
        candidates = np.where(counts == max_count)[0]
        # Deterministic tie-breaking: smallest index
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    # Ensure full coverage if greedy broke early
    if np.any(current_uncovered):
        while np.any(current_uncovered):
            uncovered_indices = np.where(current_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            best_idx = uncovered_indices[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            if time.time() > deadline - 30.0:
                break

    selected_set = set(selected)
    
    # 2. Pruning
    def prune():
        nonlocal selected, selected_set, cover_count
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            selected_set = set(new_selected)
            
    prune()
    
    best_selected = list(selected)
    best_size = len(selected)
    
    # 3. Iterated Local Search (ILS)
    # We use a deterministic seed for reproducibility
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0:
        # Perturb: Remove k random codewords and re-cover
        # k is small, e.g., 2 or 3
        k = rng.integers(1, 4)
        
        # Make a copy of state to revert if perturbation is bad?
        # Or just do it in place and revert if size increases too much?
        # Better: do it on a copy? Copying 500 ints is fast.
        
        current_selected = list(selected)
        current_set = set(current_selected)
        current_cover = cover_count.copy()
        
        # Remove k random codewords
        to_remove = rng.choice(current_selected, size=min(k, len(current_selected)), replace=False)
        for c in to_remove:
            current_cover[balls[c]] -= 1
            current_set.remove(c)
            current_selected.remove(c)
            
        # Re-cover uncovered words greedily
        uncovered = np.where(current_cover == 0)[0]
        while len(uncovered) > 0:
            # Pick one uncovered word
            # To be efficient, pick one that is covered by many candidates?
            # Or just pick the first one and add a word that covers it?
            # Greedy: for each uncovered word, we need to add a codeword.
            # Standard greedy on the remaining problem.
            
            # Let's just run a mini-greedy
            # This might be slow if we do it every iteration.
            # Optimization: Instead of full greedy, just add codewords to cover the specific holes.
            
            # Pick an uncovered word u
            u = uncovered[0]
            # Find a word w that covers u and maximizes coverage of other uncovered words
            # Candidates are B(u)
            candidates = balls[u]
            best_w = -1
            best_gain = -1
            for w in candidates:
                if w in current_set:
                    continue
                # Count how many uncovered words w covers
                gain = np.sum(current_cover[balls[w]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
            
            if best_w == -1:
                # Should not happen, but fallback
                best_w = u
                if best_w in current_set:
                     # If u is already in set (shouldn't be if cover 0), pick neighbor
                     for w in balls[u]:
                         if w not in current_set:
                             best_w = w
                             break
            
            current_set.add(best_w)
            current_selected.append(best_w)
            current_cover[balls[best_w]] += 1
            
            # Update uncovered list
            # This is O(total) if we recompute. 
            # Given total=6561, and few iterations, it's okay.
            uncovered = np.where(current_cover == 0)[0]
            
        # After re-covering, prune
        # We need a local prune function that works on the local variables
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in current_selected:
                if np.min(current_cover[balls[c]]) > 1:
                    current_cover[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            current_selected = new_selected
            current_set = set(current_selected)
            
        # Local Search: 1-Opt Swap
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(current_selected)):
                if time.time() > deadline - 5.0:
                    break
                c = current_selected[i]
                # Find critical words
                critical = np.where(current_cover[balls[c]] == 1)[0]
                if len(critical) == 0:
                    # Redundant, should have been pruned
                    current_cover[balls[c]] -= 1
                    current_set.remove(c)
                    current_selected.pop(i)
                    i -= 1 # adjust index
                    improved = True
                    continue
                
                # Find a replacement w
                # w must cover all critical words
                # w must not be in current_set
                
                # Intersection of balls of critical words
                # Since len(critical) is small (usually 1 or 2), we can iterate over B(critical[0])
                # and check if it covers others.
                
                candidates = balls[critical[0]]
                found_w = -1
                for w in candidates:
                    if w in current_set:
                        continue
                    # Check if w covers all critical words
                    ok = True
                    for u in critical:
                        if w not in balls[u]:
                            # Note: balls[u] is an array. 'w in balls[u]' is O(17).
                            # We can precompute a lookup? 
                            # For now, O(17 * len(critical)) is fine.
                            ok = False
                            break
                    if ok:
                        found_w = w
                        break
                
                if found_w != -1:
                    # Perform swap
                    current_cover[balls[c]] -= 1
                    current_set.remove(c)
                    current_selected[i] = found_w
                    current_set.add(found_w)
                    current_cover[balls[found_w]] += 1
                    improved = True
                    # Restart prune/swap loop? Or continue?
                    # Continuing might allow further swaps.
                    # But state changed, so indices might be invalid? No, list size same.
                    # However, other codewords might become removable.
                    # Let's break and restart the outer while loop to re-prune.
                    break
            
            if improved:
                # Re-prune after any swap
                changed = True
                while changed:
                    changed = False
                    new_selected = []
                    for c in current_selected:
                        if np.min(current_cover[balls[c]]) > 1:
                            current_cover[balls[c]] -= 1
                            changed = True
                        else:
                            new_selected.append(c)
                    current_selected = new_selected
                    current_set = set(current_selected)

        # Accept if better
        if len(current_selected) < best_size:
            best_size = len(current_selected)
            best_selected = list(current_selected)
            # Update global state to best
            selected = best_selected
            selected_set = set(selected)
            # Recalculate cover_count for best_selected
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1
            
            # If we hit the theoretical lower bound or known optimum, we can stop early?
            # Known optimum is 486.
            if best_size <= 486:
                break

    # Final verification
    final_cover = np.zeros(total, dtype=np.int32)
    for c in best_selected:
        final_cover[balls[c]] += 1
        
    if np.any(final_cover == 0):
        # Add missing
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            best_selected.append(m)
            final_cover[balls[m]] += 1

    # Convert to digit lists
    result = []
    for w in best_selected:
        result.append(idx_to_digits(w))
    
    return result
