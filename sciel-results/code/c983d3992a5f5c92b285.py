import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # The evaluator passes these, but we assume the standard task parameters.
    # However, the function signature is construct(q, n, radius).
    # We must handle general q, n, radius? 
    # The prompt says: "The evaluator passes this task's alphabet size q, length n and covering radius."
    # And "Score: ... upper bound on K_q(n, radius)".
    # The current seed program hardcodes n=4 logic? No, it uses n-4.
    # But the task is specifically "Football pool problem for 7 matches: ternary covering code K_3(7,1)".
    # The contract says: "Interface: construct(q, n, radius) -> list of codewords".
    # "Valid when every word of {0..q-1}^n is within Hamming distance radius of some codeword".
    # If q, n, radius are different, the problem is different.
    # However, the "Knowledge" and "Inspiration" are specific to K_3(7,1).
    # The seed program uses a construction that works for any n >= 4?
    # tetracode is length 4. It pads with product of range(3) for n-4.
    # This works for q=3.
    # If q != 3 or radius != 1, this specific optimization might not apply.
    # But the task title is "Football pool problem for 7 matches: ternary covering code K_3(7,1)".
    # It is highly likely q=3, n=7, radius=1.
    # I will write a solution that specifically targets q=3, n=7, radius=1 for speed, 
    # but falls back to a generic (slower) method or the seed if parameters differ.
    
    if q != 3 or radius != 1:
        # Fallback to seed logic or a generic greedy if parameters are different.
        # Given the specific task description, this branch is likely never taken.
        # But to be safe, let's use the seed logic for non-standard cases.
        if n >= 4:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             # This only works for q=3. If q!=3, we need a different fallback.
             # For now, assume q=3.
             if q == 3:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # If q != 3, we can't use tetracode. 
        # Let's just return a trivial valid (though large) code for non-3 cases to avoid crash.
        # A trivial code is all words.
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    total_words = 3 ** n
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin
    
    # 1. Precomputation
    # Map word (tuple of ints) to index (int)
    # Index = w0 * 3^6 + w1 * 3^5 + ... + w6
    # We need to generate all 2187 words and their indices.
    words = []
    word_to_idx = {}
    for i in range(total_words):
        # Convert i to base 3 representation of length n
        w = []
        x = i
        for _ in range(n):
            w.append(x % 3)
            x //= 3
        w = w[::-1] # reverse to get most significant first
        words.append(tuple(w))
        word_to_idx[tuple(w)] = i

    # Precompute cover masks
    # cover_mask[i] is a Python int with bit j set if word j is within distance 1 of word i.
    cover_masks = [0] * total_words
    for i in range(total_words):
        w = words[i]
        mask = 0
        # The word itself
        mask |= (1 << i)
        # Neighbors: change one position
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = w[pos] + delta
                if 0 <= new_val < 3:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_w = tuple(new_w)
                    j = word_to_idx[new_w]
                    mask |= (1 << j)
        cover_masks[i] = mask
    
    all_ones_mask = (1 << total_words) - 1
    
    def greedy_construction(randomize=False):
        uncovered = all_ones_mask
        selected = []
        while uncovered:
            best_idx = -1
            max_covered = 0
            candidates = []
            for i in range(total_words):
                # If i is already selected, skip? 
                # Actually, we can just pick the best from all. 
                # Picking an already selected one adds 0 new coverage.
                covered_count = (uncovered & cover_masks[i]).bit_count()
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_idx = i
                    candidates = [i]
                elif covered_count == max_covered and covered_count > 0:
                    candidates.append(i)
            
            if randomize and candidates:
                best_idx = random.choice(candidates)
            
            selected.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
        
        return selected

    def reduce_set(selected):
        # Remove redundant codewords
        selected = set(selected)
        changed = True
        while changed:
            changed = False
            for c in list(selected):
                # Check if c is redundant
                # Union of all other masks
                union_others = 0
                for c2 in selected:
                    if c2 != c:
                        union_others |= cover_masks[c2]
                
                # If cover_masks[c] is subset of union_others, c is redundant
                if (cover_masks[c] & ~union_others) == 0:
                    selected.remove(c)
                    changed = True
        return selected

    best_size = float('inf')
    best_set = None
    
    # Run multiple randomized greedy constructions
    num_restarts = 0
    while time.time() - start_time < time_limit * 0.5:
        selected = greedy_construction(randomize=True)
        reduced = reduce_set(selected)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
            num_restarts += 1
            # If we hit the known best (186), we can try to improve or just stop if time is tight?
            # 186 is the best known. If we hit it, we might still try to find 185? 
            # But 186 is very hard to beat.
            if best_size <= 186:
                # Try a few more to see if we can get lucky, but don't spend all time.
                if num_restarts > 20:
                    break

    if best_set is None:
        # Fallback if no time for greedy
        selected = greedy_construction(randomize=False)
        best_set = reduce_set(selected)
        best_size = len(best_set)

    # Local Search / Swap Improvement on the best set found
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Try to improve by swapping
    # We limit the number of swap attempts based on remaining time
    swap_start = time.time()
    max_swap_time = time_limit - (swap_start - start_time) - 5.0
    
    if max_swap_time > 5.0:
        improved = True
        while improved and (time.time() - swap_start) < max_swap_time:
            improved = False
            # Try swapping each codeword
            # To speed up, we can shuffle the order
            candidates_to_swap = list(current_set)
            random.shuffle(candidates_to_swap)
            
            for c_out in candidates_to_swap:
                if (time.time() - swap_start) > max_swap_time:
                    break
                
                # Remove c_out
                temp_set = current_set - {c_out}
                
                # Calculate uncovered words
                # Union of masks in temp_set
                union_temp = 0
                for c in temp_set:
                    union_temp |= cover_masks[c]
                
                uncovered = all_ones_mask & ~union_temp
                
                if uncovered == 0:
                    # c_out was redundant? reduce_set should have removed it. 
                    # But maybe it became redundant after other swaps?
                    # If so, just remove it.
                    current_set = temp_set
                    current_size -= 1
                    improved = True
                    break # Restart loop with new set
                
                # Find best c_in to cover uncovered
                best_in = -1
                max_cov = 0
                for i in range(total_words):
                    if i in temp_set:
                        continue
                    cov = (uncovered & cover_masks[i]).bit_count()
                    if cov > max_cov:
                        max_cov = cov
                        best_in = i
                
                if best_in == -1:
                    continue
                
                # Check if best_in covers ALL uncovered words
                if (uncovered & ~cover_masks[best_in]) != 0:
                    # It doesn't cover all. 
                    # We could still swap if it allows further reductions, 
                    # but that's complex. 
                    # For now, only accept if it covers all uncovered words 
                    # (which means size stays same, but might enable reduction).
                    # Actually, if it doesn't cover all, the set is invalid.
                    # So we MUST cover all uncovered words to maintain validity.
                    continue
                
                # Tentatively swap
                new_set = temp_set | {best_in}
                
                # Reduce the new set
                new_reduced = reduce_set(new_set)
                
                if len(new_reduced) < current_size:
                    current_set = new_reduced
                    current_size = len(new_reduced)
                    improved = True
                    break # Restart loop with new set
                # If len(new_reduced) == current_size, we might still accept to escape local minima?
                # But it costs time. Let's only accept strict improvements for now.
                # Or accept equal size with some probability? 
                # Given time constraints, strict improvement is safer.

    # Convert indices back to words
    result = [list(words[i]) for i in current_set]
    
    return result
