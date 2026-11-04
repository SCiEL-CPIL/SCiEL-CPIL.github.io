import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    total_words = q ** n
    
    # 1. Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # 2. Precompute neighbor indices for radius 1 (and general radius if needed, but task is radius 1)
    # For radius 1, ball size is 1 + n*(q-1)
    # We store the list of indices covered by each word
    # Using lists is faster for SA moves than bitmasks if we maintain counts, 
    # but bitmasks are good for checking validity of the whole set.
    # However, for SA, we need to evaluate the change in "uncovered count".
    # Let's use a count-based approach for the objective function during SA.
    
    # neighbors[i] = list of indices covered by codeword i
    neighbors = [[] for _ in range(total_words)]
    
    if radius == 1:
        for i in range(total_words):
            d = digits_of[i]
            neigh = [i]
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * (q ** (n - 1 - pos))
                    idx = i + delta
                    neigh.append(idx)
            neighbors[i] = neigh
    else:
        # General radius (slower, but n=8 is small)
        for i in range(total_words):
            d = digits_of[i]
            neigh = [i]
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    orig_digits = [d[p] for p in pos_comb]
                    for variation in itertools.product(range(q), repeat=r):
                        is_diff = False
                        delta = 0
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * (q ** (n - 1 - pos))
                        if not is_diff:
                            continue
                        idx = i + delta
                        neigh.append(idx)
            neighbors[i] = neigh

    # 3. Initial Solution Construction
    # Use a fast greedy to get a good starting point (valid cover)
    # Then optimize with SA.
    
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = [1] * total_words  # 1 if uncovered, 0 if covered
        code = []
        # To speed up greedy, we can use a priority queue or just iterate.
        # Given 6561 words, iterating is fine.
        
        # We maintain a list of uncovered words? No, that's expensive to update.
        # Let's just iterate all words and pick the one with max new coverage.
        # This is O(N^2 * ball_size) which might be slow for 6561.
        # 6561^2 * 17 ~ 700 million ops. Too slow in Python.
        
        # Better Greedy:
        # 1. Start with all words uncovered.
        # 2. Pick random word, cover its neighbors.
        # 3. Repeat for uncovered words?
        
        # Actually, for the seed program, greedy was fast enough?
        # The seed program used bitmasks. (1 << 6561) is a big int.
        # Python big int bitwise ops are very fast (C implementation).
        # Let's stick to bitmasks for the initial greedy to get a valid cover quickly.
        
        balls = [0] * total_words
        for i in range(total_words):
            mask = 0
            for idx in neighbors[i]:
                mask |= (1 << idx)
            balls[i] = mask
            
        all_mask = (1 << total_words) - 1
        uncovered = all_mask
        code = []
        
        while uncovered:
            if time.time() > deadline - 20.0:
                return code # Return partial if timeout, though unlikely for greedy
            best_idx = -1
            max_cover = 0
            candidates = []
            
            for i in range(total_words):
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
                    
            if max_cover == 0:
                break
                
            if len(candidates) > 1:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
                
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    # Get initial code
    initial_code = greedy_construct(42)
    
    # 4. Simulated Annealing
    # Objective: Minimize (len(code) * 1000 + uncovered_count)
    # We want to reduce len(code) primarily, but allow uncovered to increase temporarily.
    
    # Data structures for SA:
    # code_set: set of indices in current code
    # cover_count[i]: number of codewords in code that cover word i
    # uncovered_count: number of words with cover_count[i] == 0
    
    code_set = set(initial_code)
    cover_count = [0] * total_words
    for c in code_set:
        for idx in neighbors[c]:
            cover_count[idx] += 1
            
    uncovered_count = sum(1 for c in cover_count if c == 0)
    
    def get_energy():
        return len(code_set) * 1000 + uncovered_count
        
    current_energy = get_energy()
    best_code = list(code_set)
    best_energy = current_energy
    
    T = 100.0 # Initial temperature
    cooling_rate = 0.995
    min_T = 0.1
    
    # To speed up SA, we need efficient add/remove.
    # Add codeword c:
    #   for idx in neighbors[c]:
    #       if cover_count[idx] == 0: uncovered_count -= 1
    #       cover_count[idx] += 1
    #   code_set.add(c)
    #   energy_change = 1000 - (reduction in uncovered)
    
    # Remove codeword c:
    #   for idx in neighbors[c]:
    #       cover_count[idx] -= 1
    #       if cover_count[idx] == 0: uncovered_count += 1
    #   code_set.remove(c)
    #   energy_change = -1000 + (increase in uncovered)
    
    # We will perform moves:
    # 1. Add a random word not in code.
    # 2. Remove a random word from code.
    # 3. Swap a word in code with a word not in code.
    
    # Precompute list of all indices for random selection
    all_indices = list(range(total_words))
    
    steps = 0
    max_steps = 50000 # Limit steps to ensure we finish
    
    while T > min_T and time.time() < deadline - 5.0:
        steps += 1
        if steps > max_steps:
            break
            
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            if len(code_set) >= total_words:
                continue
            # Pick random not in code
            # Using a set for membership check is O(1)
            # Picking random from all_indices and checking is O(1) avg
            while True:
                c = random.choice(all_indices)
                if c not in code_set:
                    break
            
            # Calculate delta
            delta_uncovered = 0
            for idx in neighbors[c]:
                if cover_count[idx] == 0:
                    delta_uncovered -= 1
            
            delta_energy = 1000 - delta_uncovered
            
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                # Accept
                code_set.add(c)
                for idx in neighbors[c]:
                    cover_count[idx] += 1
                    if cover_count[idx] == 1:
                        uncovered_count -= 1
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code = list(code_set)
                    
        elif move_type < 0.7:
            # Remove
            if not code_set:
                continue
            c = random.choice(list(code_set))
            
            delta_uncovered = 0
            for idx in neighbors[c]:
                if cover_count[idx] == 1:
                    delta_uncovered += 1
            
            delta_energy = -1000 + delta_uncovered
            
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                # Accept
                code_set.remove(c)
                for idx in neighbors[c]:
                    cover_count[idx] -= 1
                    if cover_count[idx] == 0:
                        uncovered_count += 1
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code = list(code_set)
                    
        else:
            # Swap
            if not code_set or len(code_set) >= total_words:
                continue
            c_old = random.choice(list(code_set))
            while True:
                c_new = random.choice(all_indices)
                if c_new not in code_set:
                    break
            
            # Calculate delta for removing c_old and adding c_new
            # Note: c_old and c_new might cover overlapping sets.
            # We must calculate the net change carefully.
            
            # Temporarily remove c_old
            # We can just simulate the changes.
            
            # It's safer to do it in two steps or calculate combined delta.
            # Combined delta:
            # 1. Remove c_old
            # 2. Add c_new
            
            # Let's just do the actual updates and rollback if rejected.
            # This is slower but less prone to error in delta calculation.
            
            # However, rollback is expensive.
            # Let's calculate delta properly.
            
            # change in cover_count for each idx in N(c_old) U N(c_new)
            
            # For idx in N(c_old) \ N(c_new): count decreases by 1
            # For idx in N(c_new) \ N(c_old): count increases by 1
            # For idx in N(c_old) \cap N(c_new): count unchanged
            
            # We need to know how many of these changes cross the 0/1 threshold.
            
            # This is complex to do without iterating.
            # Given ball size is 17, iterating is fast.
            
            old_covered_uncovered = 0
            new_covered_uncovered = 0
            
            # Simulate removal of c_old
            # We need to know which indices become uncovered.
            # An index becomes uncovered if cover_count[idx] == 1 and idx is NOT covered by c_new.
            
            # Simulate addition of c_new
            # An index becomes covered if cover_count[idx] == 0 (after removal) and idx is covered by c_new.
            
            # This is tricky. Let's just do the "Apply and Check" method for Swap.
            # It's only 30% of moves.
            
            # Apply removal
            for idx in neighbors[c_old]:
                cover_count[idx] -= 1
            # Apply addition
            for idx in neighbors[c_new]:
                cover_count[idx] += 1
            
            # Recalculate uncovered count? No, we updated cover_count.
            # We can just recalculate uncovered_count from scratch? No, O(N).
            # We can track the change in uncovered_count during the update.
            
            # Let's restart the swap logic with proper delta tracking.
            
            # Rollback
            for idx in neighbors[c_old]:
                cover_count[idx] += 1
            for idx in neighbors[c_new]:
                cover_count[idx] -= 1
            
            # Okay, let's just skip Swap for now to keep it simple and robust, 
            # or implement it carefully.
            # Actually, Add/Remove is sufficient to explore the space.
            # Add increases size, Remove decreases.
            # The energy landscape will guide it.
            
            # Let's just do Add/Remove. It's safer.
            pass
            
        T *= cooling_rate
        
    # Final Local Search: Greedy Removal
    # Try to remove any codeword that doesn't increase uncovered count.
    # This ensures we end with a valid cover if possible, or minimal uncovered.
    
    # Sort code by "usefulness" (number of uniquely covered words) to remove bad ones first?
    # Or just random order.
    
    # Let's do a few passes of greedy removal on the best_code found by SA.
    # But we want to return the best valid cover.
    # If best_code from SA is invalid, we might need to fix it or use the initial valid one.
    # The energy function penalizes uncovered words heavily (1000 per word vs 1 per codeword).
    # So SA should strongly prefer valid covers.
    
    # Let's run a local search on best_code to clean up.
    # We want to minimize size while keeping uncovered == 0.
    
    final_code = best_code[:]
    
    # Rebuild cover counts for final_code
    cover_count_final = [0] * total_words
    for c in final_code:
        for idx in neighbors[c]:
            cover_count_final[idx] += 1
            
    uncovered_final = sum(1 for c in cover_count_final if c == 0)
    
    # If it's not valid, we might have a problem. 
    # But with weight 1000, it's very unlikely to stay invalid if a valid solution exists nearby.
    # If it is invalid, we should probably return the initial valid greedy solution if it's smaller?
    # Or just return the SA result and hope the evaluator accepts "best effort"? 
    # No, the task requires "Valid when every word... is within distance".
    # So we MUST return a valid cover.
    
    # Strategy: 
    # 1. Run SA.
    # 2. If best_code is valid, use it.
    # 3. If not, use the initial greedy code (which is valid).
    # 4. Run local search (remove redundant) on the chosen valid code.
    
    # Check validity of best_code
    if uncovered_final > 0:
        # Fallback to initial valid code
        final_code = initial_code[:]
        cover_count_final = [0] * total_words
        for c in final_code:
            for idx in neighbors[c]:
                cover_count_final[idx] += 1
        uncovered_final = sum(1 for c in cover_count_final if c == 0)
        
    # Local Search: Remove redundant
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Shuffle to avoid bias
        random.shuffle(final_code)
        for i in range(len(final_code)):
            c = final_code[i]
            # Check if removing c keeps cover valid
            # i.e., no index in neighbors[c] has cover_count == 1
            can_remove = True
            for idx in neighbors[c]:
                if cover_count_final[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                final_code.pop(i)
                for idx in neighbors[c]:
                    cover_count_final[idx] -= 1
                improved = True
                break # Restart loop
            
    # Convert indices to lists of digits
    return [list(digits_of[i]) for i in final_code]
