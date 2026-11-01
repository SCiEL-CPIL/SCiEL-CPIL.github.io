import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_iterated(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_iterated(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls for all words
    # balls[w, 0] = w, balls[w, 1:17] = neighbors
    # Using big-endian weights for index calculation
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
                
    best_size = total
    best_selected = None
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in the search process
    
    # We will run multiple iterations of Randomized Greedy + Local Search
    while time.time() < deadline - 10.0:
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        current_uncovered = np.ones(total, dtype=bool)
        
        # Greedy Loop with Randomization
        # Instead of always picking the max, pick from top candidates to diversify
        while np.any(current_uncovered):
            if time.time() > deadline - 15.0:
                break # Time out for this iteration, skip to next if valid, else break outer
            
            # Count uncovered neighbors for each word
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            # Get candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best candidates
            # This is the key difference from pure deterministic greedy
            best_idx = candidates[rng.integers(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
        
        # If timed out during greedy, we might have an invalid partial solution.
        # Complete it quickly to ensure validity for the local search phase.
        if np.any(current_uncovered):
            if time.time() > deadline - 10.0:
                break
            # Simple completion: pick any uncovered word
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
        
        # 2. Aggressive Pruning (Greedy Removal)
        # Remove codewords that are no longer necessary
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if removing c is safe: all words in its ball are covered by others
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # 3. Local Search: Swap-based improvement (2-Opt style)
        # Try to replace a codeword with another to allow further removals
        # This is the "polish" step.
        
        improved = True
        max_swap_iters = 5
        swap_iter = 0
        while improved and time.time() < deadline - 10.0 and swap_iter < max_swap_iters:
            improved = False
            swap_iter += 1
            new_selected = list(selected)
            new_selected_set = set(new_selected)
            
            for i in range(len(new_selected)):
                if time.time() > deadline - 10.0:
                    break
                c = new_selected[i]
                
                # Try to replace c with a neighbor w
                # We iterate through neighbors of c. 
                # Optimization: Only consider neighbors that are not already in selected.
                for w in balls[c]:
                    if w in new_selected_set:
                        continue
                    
                    # Candidate swap: remove c, add w
                    # 1. Remove c
                    cover_count[balls[c]] -= 1
                    
                    # Check if any word is now uncovered
                    if np.any(cover_count == 0):
                        cover_count[balls[c]] += 1 # Restore
                        continue
                    
                    # 2. Add w
                    cover_count[balls[w]] += 1
                    
                    # 3. Check if we can remove another codeword d (d != c, d != w)
                    # We look for a d in new_selected (excluding c) such that min(cover_count[balls[d]]) > 1
                    
                    removable_d = None
                    # Optimization: Only check codewords that might be affected?
                    # For simplicity and robustness, we check all, but break early.
                    # N is small (~500), so O(N) is fine.
                    
                    for j in range(len(new_selected)):
                        if j == i: # c
                            continue
                        d = new_selected[j]
                        if np.min(cover_count[balls[d]]) > 1:
                            removable_d = d
                            break
                    
                    if removable_d is not None:
                        # Perform the swap: Remove c, Add w, Remove d
                        # We already did: Remove c, Add w
                        # Now Remove d
                        cover_count[balls[removable_d]] -= 1
                        
                        # Update selected list
                        new_selected[i] = w # Replace c with w
                        new_selected_set.remove(c)
                        new_selected_set.add(w)
                        
                        new_selected.remove(removable_d)
                        new_selected_set.remove(removable_d)
                        
                        improved = True
                        break
                    else:
                        # Swap didn't help remove another node.
                        # Restore state
                        cover_count[balls[w]] -= 1
                        cover_count[balls[c]] += 1
                    
                    if improved:
                        break
                
                if improved:
                    # Reset loop to restart with new state
                    selected = new_selected
                    new_selected_set = new_selected_set # Just for clarity, it's updated inside
                    break
            
            # If no improvement in this pass, break
            if not improved:
                break

        # 4. Final Pruning Pass after swaps
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

        # Update Best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected.copy()
            
            if best_size <= 486:
                # We hit the known optimum (or close to it). 
                # 486 is the best known. If we get 486, we can stop early to save time?
                # Actually, we might get 485? Unlikely. 
                # Let's stop if we reach 486 to be safe and fast.
                break

    # Fallback if best_selected is None (should not happen if loop runs at least once)
    if best_selected is None:
        # Return a valid but large set
        best_selected = list(range(total))
        
    # Convert to digit lists
    result = []
    for w in best_selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
