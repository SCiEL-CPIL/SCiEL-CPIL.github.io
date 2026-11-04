import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    if radius != 1 or q != 3 or n != 8:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Parameters
    total_words = q ** n
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Precompute spheres as bitmasks
    # Map word to index
    word_to_idx = {}
    all_words = []
    for i in range(total_words):
        w = [int(d) for d in format(i, '0{}b'.format(n)).zfill(n)] # This is base 2, need base 3
        # Better: use itertools
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Create bitmask for each word's sphere
    sphere_masks = [0] * total_words
    for w in all_words:
        idx = word_to_idx[w]
        mask = 0
        # Center
        mask |= (1 << idx)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    n_idx = word_to_idx[tuple(new_w)]
                    mask |= (1 << n_idx)
        sphere_masks[idx] = mask
        
    # Initialize random solution
    # Heuristic start: Random 500 words
    initial_size = 500
    current_code_indices = set(random.sample(range(total_words), initial_size))
    
    # Calculate uncovered
    covered_mask = 0
    for idx in current_code_indices:
        covered_mask |= sphere_masks[idx]
    uncovered_mask = ((1 << total_words) - 1) & ~covered_mask
    
    best_code_indices = set(current_code_indices)
    best_uncovered = uncovered_mask
    best_size = len(current_code_indices)
    
    # Local Search
    # We will try to remove codewords and add codewords
    # State: current_code_indices, uncovered_mask
    
    # Helper to update uncovered when removing/adding
    # For efficiency, we recompute uncovered from scratch periodically or incrementally.
    # Incremental:
    # When removing i: new_uncovered = uncovered | (sphere_mask[i] & ~covered_by_others)
    # This is hard to track "covered_by_others" without recounting.
    # Simpler: Recalculate uncovered from scratch for every move? 
    # 6561 bits is small. `covered_mask = 0; for i in code: covered_mask |= mask[i]`
    # If code size is ~400, this is 400 OR operations on big ints. Very fast in Python.
    
    def calc_uncovered(code_set):
        cov = 0
        for i in code_set:
            cov |= sphere_masks[i]
        return ((1 << total_words) - 1) & ~cov

    current_uncovered = calc_uncovered(current_code_indices)
    
    # Simulated Annealing / Random Walk
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Try a random move
        # 50% remove, 50% add
        if random.random() < 0.5 and len(current_code_indices) > 100:
            # Remove
            idx_to_remove = random.choice(list(current_code_indices))
            new_code = set(current_code_indices)
            new_code.remove(idx_to_remove)
            new_uncovered = calc_uncovered(new_code)
            
            # Cost difference: change in number of uncovered words
            old_uncovered_count = bin(current_uncovered).count('1')
            new_uncovered_count = bin(new_uncovered).count('1')
            delta_cost = new_uncovered_count - old_uncovered_count
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                current_code_indices = new_code
                current_uncovered = new_uncovered
                if len(current_code_indices) < best_size:
                    best_size = len(current_code_indices)
                    best_code_indices = set(current_code_indices)
                    best_uncovered = current_uncovered
        else:
            # Add
            # Pick a random word, or a word that covers many uncovered
            # For speed, pick random
            idx_to_add = random.randint(0, total_words - 1)
            if idx_to_add not in current_code_indices:
                new_code = set(current_code_indices)
                new_code.add(idx_to_add)
                new_uncovered = calc_uncovered(new_code)
                
                old_uncovered_count = bin(current_uncovered).count('1')
                new_uncovered_count = bin(new_uncovered).count('1')
                delta_cost = new_uncovered_count - old_uncovered_count
                # Note: Adding increases size, so we want to minimize size.
                # If it covers uncovered words, delta_cost is negative (good).
                # But we also pay +1 for size.
                # Objective: Minimize Size + Penalty * Uncovered
                # Let's use a combined cost: Size + 100 * Uncovered_count
                # Actually, we want to reach 0 uncovered first, then minimize size.
                
                # Let's stick to: Accept if it reduces uncovered, or if it keeps uncovered same and reduces size (impossible when adding).
                # So adding is only useful if it reduces uncovered.
                if delta_cost < 0 or (delta_cost == 0 and random.random() < 0.1): # Tie break
                    current_code_indices = new_code
                    current_uncovered = new_uncovered
                    if len(current_code_indices) < best_size and bin(best_uncovered).count('1') >= bin(current_uncovered).count('1'):
                        # Only update best if we are not worse on coverage
                        # Actually, we should track best valid solution.
                        # If current_uncovered is 0, it's a valid solution.
                        if bin(current_uncovered).count('1') == 0:
                            best_size = len(current_code_indices)
                            best_code_indices = set(current_code_indices)
                            best_uncovered = current_uncovered
        
        T *= cooling_rate
        
    # Ensure we return a valid solution.
    # If best_uncovered is not 0, we need to fix it with a greedy pass on the remaining uncovered.
    if bin(best_uncovered).count('1') > 0:
        # Greedy cover for the remaining holes
        temp_code = set(best_code_indices)
        temp_uncovered = best_uncovered
        while temp_uncovered:
            # Find best word to add
            best_idx = -1
            best_gain = 0
            # Sample a few candidates to save time? Or check all?
            # Checking all 6561 is fast with bitsets.
            for i in range(total_words):
                if i in temp_code:
                    continue
                gain = bin(sphere_masks[i] & temp_uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            temp_code.add(best_idx)
            temp_uncovered &= ~sphere_masks[best_idx]
        best_code_indices = temp_code
        best_uncovered = temp_uncovered
        
    # If still invalid (shouldn't happen), fallback to greedy from scratch
    if bin(best_uncovered).count('1') > 0:
        # Re-run simple greedy
        uncovered = ((1 << total_words) - 1)
        code = set()
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(total_words):
                if i in code:
                    continue
                gain = bin(sphere_masks[i] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            code.add(best_idx)
            uncovered &= ~sphere_masks[best_idx]
        best_code_indices = code
        best_uncovered = uncovered

    return [list(all_words[i]) for i in best_code_indices]
