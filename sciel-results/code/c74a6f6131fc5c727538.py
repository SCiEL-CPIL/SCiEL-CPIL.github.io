import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute ball masks for each word
    # Ball of radius 1 around w includes w and all words at distance 1.
    ball_masks = [0] * total_words
    
    for i, w in enumerate(all_words):
        mask = 1 << i  # Include self
        
        # Generate distance 1 neighbors
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                # Create new word
                new_word = w[:pos] + (new_val,) + w[pos+1:]
                j = word_to_idx[new_word]
                mask |= (1 << j)
                
        ball_masks[i] = mask

    # Helper to get index from word
    def get_idx(word):
        return word_to_idx[word]

    # 1. Greedy Initialization
    # We want to start with a small valid set.
    # Greedy Set Cover
    uncovered = (1 << total_words) - 1
    code = []
    code_set = set()
    
    # Symmetry breaking: Fix first codeword to all zeros
    first_word = (0,) * n
    first_idx = get_idx(first_word)
    code.append(first_word)
    code_set.add(first_word)
    uncovered &= ~ball_masks[first_idx]
    
    while uncovered != 0 and time.time() < deadline - 10.0: # Leave time for SA
        best_idx = -1
        best_count = 0
        
        # Find the word that covers the most uncovered words
        # Optimization: Only check candidates that are not in code?
        # Actually, checking all is O(N^2). N=729. 729^2 ~ 500k.
        # We do this until covered.
        # We can speed up by only checking words that cover at least one uncovered word.
        # But simple loop is fine for 729.
        
        for i in range(total_words):
            if all_words[i] in code_set:
                continue
            covered = uncovered & ball_masks[i]
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_idx == -1:
            break
            
        best_word = all_words[best_idx]
        code.append(best_word)
        code_set.add(best_word)
        uncovered &= ~ball_masks[best_idx]

    # Initial size
    current_size = len(code)
    
    # 2. Simulated Annealing
    # Objective: Minimize Size + Penalty * Uncovered
    # Penalty should be large enough to prefer valid solutions.
    # If we have size N and 0 uncovered, cost = N.
    # If we have size N-1 and 1 uncovered, cost = N-1 + Penalty.
    # We want N-1+Penalty > N, so Penalty > 1. Let's use Penalty = 100.
    
    PENALTY = 100.0
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Track best solution found
    best_code = list(code)
    best_size = current_size
    
    # Precompute list of indices for random selection
    all_indices = list(range(total_words))
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Choose a move type
        # 1. Remove (prob 0.5)
        # 2. Swap (prob 0.5)
        
        if not code:
            break
            
        if random.random() < 0.5:
            # Remove move
            idx_to_remove = random.randint(0, len(code) - 1)
            word_to_remove = code[idx_to_remove]
            mask_to_remove = ball_masks[get_idx(word_to_remove)]
            
            # Calculate new uncovered
            # We need to know which words are ONLY covered by this word.
            # Efficient way: 
            # Current coverage is full (assuming we start valid and only accept valid or penalized).
            # Actually, let's maintain a coverage count array? 
            # With 729 words, maintaining an array of 729 ints is cheap.
            # Let's recompute uncovered from scratch for the "trial" state?
            # Trial state: code without word_to_remove.
            # New uncovered = Union of balls of (code - {word_to_remove}) complement.
            
            # To be fast, let's calculate the set of words covered by code without this one.
            # covered_union = 0
            # for w in code:
            #    if w != word_to_remove:
            #       covered_union |= ball_masks[get_idx(w)]
            # new_uncovered = ~covered_union & ((1 << total_words) - 1)
            
            # This is O(N_code). N_code ~ 73-80.
            # 80 * (bit ops on 729 bits) is very fast.
            
            covered_union = 0
            for w in code:
                if w == word_to_remove:
                    continue
                covered_union |= ball_masks[get_idx(w)]
                
            new_uncovered = ~covered_union & ((1 << total_words) - 1)
            num_uncovered = new_uncovered.bit_count()
            
            new_size = len(code) - 1
            new_cost = new_size + PENALTY * num_uncovered
            old_cost = len(code) + PENALTY * 0 # Assuming current state is valid (0 uncovered)
            
            delta_cost = new_cost - old_cost
            
            if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                code.pop(idx_to_remove)
                code_set.discard(word_to_remove)
                
                if num_uncovered == 0:
                    # Valid solution
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                        
        else:
            # Swap move
            idx_to_swap = random.randint(0, len(code) - 1)
            old_word = code[idx_to_swap]
            old_mask = ball_masks[get_idx(old_word)]
            
            # Pick a random new word
            new_word_idx = random.choice(all_indices)
            new_word = all_words[new_word_idx]
            
            if new_word in code_set:
                continue
                
            new_mask = ball_masks[new_word_idx]
            
            # Calculate new uncovered
            # Remove old, add new
            covered_union = 0
            for w in code:
                if w == old_word:
                    continue
                covered_union |= ball_masks[get_idx(w)]
            covered_union |= new_mask
            
            new_uncovered = ~covered_union & ((1 << total_words) - 1)
            num_uncovered = new_uncovered.bit_count()
            
            new_size = len(code)
            new_cost = new_size + PENALTY * num_uncovered
            old_cost = len(code) + PENALTY * 0
            
            delta_cost = new_cost - old_cost
            
            if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                code[idx_to_swap] = new_word
                code_set.discard(old_word)
                code_set.add(new_word)
                
                if num_uncovered == 0:
                    # Valid solution, size didn't change, but structure did.
                    # This might enable future removals.
                    # We don't update best_code here as size is same, 
                    # unless we want to keep the "best" valid configuration?
                    # No, we only care about size.
                    pass
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final local search on best_code to try and remove more?
    # The SA might have found a valid code of size best_size.
    # Let's do a quick greedy removal pass on best_code to ensure it's minimal.
    
    final_code = list(best_code)
    final_set = set(final_code)
    
    # Greedy removal
    improved = True
    while improved:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            w = final_code[i]
            mask = ball_masks[get_idx(w)]
            
            # Check if removing w leaves any word uncovered
            # Calculate coverage of final_code without w
            covered_union = 0
            for v in final_code:
                if v == w:
                    continue
                covered_union |= ball_masks[get_idx(v)]
                
            if covered_union == (1 << total_words) - 1:
                # Can remove
                final_code.pop(i)
                final_set.discard(w)
                improved = True
                break # Restart loop as indices changed
                
    return [list(w) for w in final_code]
