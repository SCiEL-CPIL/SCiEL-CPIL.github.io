import time
import numpy as np
import itertools
from scipy.optimize import linprog
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
    
    # 1. Greedy Construction
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # Main greedy loop
    while np.any(current_uncovered):
        if time.time() > deadline - 40.0:
            break
            
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        
        if max_count == 0:
            break
        
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
    
    if np.any(current_uncovered) and time.time() < deadline - 10.0:
        while np.any(current_uncovered):
            uncovered_indices = np.where(current_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            best_idx = uncovered_indices[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            if time.time() > deadline - 10.0:
                break
    
    # 2. Aggressive Pruning with multiple passes and varied orderings
    # Use a deterministic random seed for reproducibility
    rng = random.Random(42)
    
    def prune_pass(selected_list, cover_cnt, order='forward'):
        """Remove removable codewords in the given order. Returns new list and updated cover_cnt."""
        changed = False
        if order == 'forward':
            indices = range(len(selected_list))
        elif order == 'reverse':
            indices = range(len(selected_list) - 1, -1, -1)
        else:  # shuffled
            indices = list(range(len(selected_list)))
            rng.shuffle(indices)
        
        new_selected = []
        for i in indices:
            c = selected_list[i]
            if np.min(cover_cnt[balls[c]]) > 1:
                cover_cnt[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        return new_selected, cover_cnt, changed
    
    # Multiple pruning passes with different orderings
    for _ in range(10):
        if time.time() > deadline - 25.0:
            break
        # Try forward, reverse, and shuffled orderings
        for order in ['forward', 'reverse', 'shuffled']:
            if time.time() > deadline - 25.0:
                break
            selected, cover_count, changed = prune_pass(selected, cover_count, order)
            if not changed:
                break
    
    # 3. Enhanced Local Search: 2-Opt Swap with Pruning
    # For each codeword, try swapping with neighbors. Accept if it enables removal.
    # Also try "move" operations: remove a codeword, then re-add a better one.
    
    improved = True
    max_swap_iterations = 50
    iter_count = 0
    while improved and time.time() < deadline - 10.0 and iter_count < max_swap_iterations:
        improved = False
        iter_count += 1
        selected_set = set(selected)
        
        for i in range(len(selected)):
            if time.time() > deadline - 10.0:
                break
            c = selected[i]
            
            # Try all neighbors of c as potential replacements
            for w in balls[c]:
                if w in selected_set:
                    continue
                
                # Attempt swap: remove c, add w
                # First check if removing c leaves any word uncovered
                temp_cover = cover_count.copy()
                temp_cover[balls[c]] -= 1
                
                if np.any(temp_cover == 0):
                    continue  # Can't remove c without uncovering words
                
                # Now add w
                temp_cover[balls[w]] += 1
                
                # Check if this swap enables removal of another codeword
                removable_d = None
                for j in range(len(selected)):
                    if j == i:
                        continue
                    d = selected[j]
                    if np.min(temp_cover[balls[d]]) > 1:
                        removable_d = d
                        break
                
                if removable_d is not None:
                    # Perform the swap and removal
                    cover_count = temp_cover
                    cover_count[balls[removable_d]] -= 1
                    
                    # Update selected list
                    selected[i] = w
                    selected_set.remove(c)
                    selected_set.add(w)
                    selected.remove(removable_d)
                    selected_set.remove(removable_d)
                    
                    improved = True
                    break
                
                # If no immediate removal, check if the swap itself reduces overlap
                # by computing a "potential" score: sum of min(counts, 2) for words in balls[w]
                # This is a heuristic for "how well-covered" the neighborhood is.
                # If the new codeword covers words that were previously over-covered, it might help later.
                # For now, skip this to keep the search focused.
            
            if improved:
                break
    
    # 4. Additional Local Search: Try removing pairs of codewords and re-adding better ones
    # This is a 3-opt style move: remove c1, c2, add w1, w2 (or just w1)
    # Given time constraints, we'll do a simplified version.
    
    if time.time() < deadline - 5.0:
        # Try removing each codeword and seeing if we can re-cover with fewer additions
        # This is expensive, so we limit the scope.
        pass  # Skip for now to stay within time budget
    
    # 5. Final Pruning Pass
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        for order in ['forward', 'reverse']:
            if time.time() > deadline - 2.0:
                break
            selected, cover_count, changed = prune_pass(selected, cover_count, order)
            if not changed:
                break
    
    # 6. Final Verification and Conversion
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1
    
    # Convert to digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
