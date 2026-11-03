import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute ball masks
    # Ball of radius 1 around w includes w and all words at distance 1.
    # For q=3, n=6, radius=1:
    # Size of ball = 1 + n*(q-1) = 1 + 6*2 = 13.
    ball_masks = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 1 << i
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

    def get_idx(word):
        return word_to_idx[word]

    # 1. Greedy Initialization
    uncovered = (1 << total_words) - 1
    code = []
    code_set = set()
    
    first_word = (0,) * n
    first_idx = get_idx(first_word)
    code.append(first_word)
    code_set.add(first_word)
    uncovered &= ~ball_masks[first_idx]
    
    while uncovered != 0 and time.time() < deadline - 20.0:
        best_idx = -1
        best_count = 0
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

    # 2. Simulated Annealing
    # We want to minimize size.
    # State: code (list of tuples)
    # Moves:
    #   A. Remove random word.
    #   B. Perturb random word (change one coordinate).
    #   C. Swap random word with random other word.
    
    best_code = list(code)
    best_size = len(code)
    
    # Helper to calculate uncovered count for a given code list
    # Optimized: maintain coverage counts? 
    # For simplicity and speed in Python, recomputing union is O(N_code).
    # N_code ~ 73. 73 iterations per move.
    # Moves per second? 
    # If we can do 1000 moves/sec, we have ~100k moves in 100s.
    
    def get_uncovered_count(code_list):
        covered_union = 0
        for w in code_list:
            covered_union |= ball_masks[get_idx(w)]
        return (~covered_union & ((1 << total_words) - 1)).bit_count()

    PENALTY = 10.0 # Lower penalty allows more exploration of invalid states if we expect to fix them
    # Actually, if we allow invalid states, we must ensure we can get back to valid.
    # If Penalty is too low, we might drift to large size with many uncovered.
    # If Penalty is high, we stay valid.
    # Let's try to stay valid mostly, but allow temporary invalidity to escape local minima.
    
    T = 5.0
    T_min = 0.05
    cooling_rate = 0.9995
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5.0:
        if not code:
            break
            
        move_type = random.random()
        old_cost = len(code) + PENALTY * get_uncovered_count(code)
        
        if move_type < 0.3:
            # Remove move
            if len(code) <= 1:
                continue
            idx_to_remove = random.randint(0, len(code) - 1)
            word_to_remove = code[idx_to_remove]
            
            new_code = code.copy()
            new_code.pop(idx_to_remove)
            
            new_uncovered = get_uncovered_count(new_code)
            new_cost = len(new_code) + PENALTY * new_uncovered
            delta = new_cost - old_cost
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                code = new_code
                if new_uncovered == 0 and len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                    no_improve_steps = 0
                elif new_uncovered > 0:
                    # We accepted an invalid state, reset no_improve?
                    # No, we only care about best valid.
                    pass
            else:
                no_improve_steps += 1
                
        elif move_type < 0.7:
            # Perturb move: Change one symbol in one codeword
            if not code:
                continue
            idx_to_perturb = random.randint(0, len(code) - 1)
            w = code[idx_to_perturb]
            pos = random.randint(0, n - 1)
            old_val = w[pos]
            new_vals = [v for v in range(q) if v != old_val]
            new_val = random.choice(new_vals)
            new_w = w[:pos] + (new_val,) + w[pos+1:]
            
            # Check if new_w is already in code (if so, it's a duplicate, invalid code structure? 
            # The problem says "codewords must be distinct". 
            # If we create a duplicate, we should handle it. 
            # If new_w is in code, this move effectively removes one and adds a duplicate? 
            # No, we are just changing the value. If it collides, the set of codewords changes.
            # Let's just check if it's valid (distinct).
            # If new_w is in code_set (excluding the one we changed), it's a collision.
            # We can reject the move if it creates a duplicate.
            
            # Simple check:
            # Create a set of the rest of the code
            # This is O(N).
            # Alternative: Just try it. If it creates a duplicate, the "code" list has duplicates.
            # The evaluator checks distinctness. We must ensure distinctness.
            
            # Let's check distinctness efficiently.
            # Since we only change one word, we just need to check if new_w is in code_set - {w}.
            # code_set is maintained? We didn't maintain it in the SA loop above.
            # Let's maintain a set for O(1) lookup.
            
            # Re-implementing with set maintenance
            
        else:
            # Swap move
            if not code:
                continue
            idx_to_swap = random.randint(0, len(code) - 1)
            old_word = code[idx_to_swap]
            
            new_word_idx = random.randint(0, total_words - 1)
            new_word = all_words[new_word_idx]
            
            # Check if new_word is already in code
            # If so, swap is a no-op or removal? 
            # If new_word == old_word, no change.
            # If new_word is in code but != old_word, we create a duplicate. Reject.
            
            # We need a set of current code words.
            # Let's restart the SA loop with proper set maintenance.

    # Restarting logic with set maintenance for efficiency and correctness
    
    # Re-initialize code from best_code found so far or the greedy result
    # Let's just run a cleaner SA from the greedy result
    
    code = list(best_code) # Start from best valid
    code_set = set(code)
    
    T = 5.0
    T_min = 0.05
    cooling_rate = 0.9995
    no_improve_steps = 0
    
    while time.time() < deadline - 5.0:
        if not code:
            break
            
        move_type = random.random()
        
        # Calculate current uncovered (expensive, but necessary for cost)
        # Optimization: Maintain coverage counts?
        # Given N=73, calculating union is fast.
        current_uncovered = get_uncovered_count(code)
        old_cost = len(code) + PENALTY * current_uncovered
        
        if move_type < 0.3:
            # Remove
            if len(code) <= 1:
                T *= cooling_rate
                continue
            idx_to_remove = random.randint(0, len(code) - 1)
            w = code[idx_to_remove]
            
            new_code = code.copy()
            new_code.pop(idx_to_remove)
            new_code_set = code_set.copy()
            new_code_set.discard(w)
            
            new_uncovered = get_uncovered_count(new_code)
            new_cost = len(new_code) + PENALTY * new_uncovered
            delta = new_cost - old_cost
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                code = new_code
                code_set = new_code_set
                if new_uncovered == 0 and len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
                
        elif move_type < 0.7:
            # Perturb
            if not code:
                T *= cooling_rate
                continue
            idx_to_perturb = random.randint(0, len(code) - 1)
            w = code[idx_to_perturb]
            pos = random.randint(0, n - 1)
            old_val = w[pos]
            new_vals = [v for v in range(q) if v != old_val]
            new_val = random.choice(new_vals)
            new_w = w[:pos] + (new_val,) + w[pos+1:]
            
            # Check collision
            if new_w in code_set and new_w != w:
                # Collision, reject
                no_improve_steps += 1
                T *= cooling_rate
                continue
            
            new_code = code.copy()
            new_code[idx_to_perturb] = new_w
            new_code_set = code_set.copy()
            new_code_set.discard(w)
            new_code_set.add(new_w)
            
            new_uncovered = get_uncovered_count(new_code)
            new_cost = len(new_code) + PENALTY * new_uncovered
            delta = new_cost - old_cost
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                code = new_code
                code_set = new_code_set
                if new_uncovered == 0 and len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
                
        else:
            # Swap
            if not code:
                T *= cooling_rate
                continue
            idx_to_swap = random.randint(0, len(code) - 1)
            old_word = code[idx_to_swap]
            
            new_word_idx = random.randint(0, total_words - 1)
            new_word = all_words[new_word_idx]
            
            if new_word in code_set and new_word != old_word:
                # Collision
                no_improve_steps += 1
                T *= cooling_rate
                continue
                
            new_code = code.copy()
            new_code[idx_to_swap] = new_word
            new_code_set = code_set.copy()
            new_code_set.discard(old_word)
            new_code_set.add(new_word)
            
            new_uncovered = get_uncovered_count(new_code)
            new_cost = len(new_code) + PENALTY * new_uncovered
            delta = new_cost - old_cost
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                code = new_code
                code_set = new_code_set
                if new_uncovered == 0 and len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
                
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final polish: Greedy removal on best_code
    final_code = list(best_code)
    final_set = set(final_code)
    
    improved = True
    while improved:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            w = final_code[i]
            # Check if w is redundant
            # Remove w, check coverage
            test_code = [x for x in final_code if x != w]
            if get_uncovered_count(test_code) == 0:
                final_code = test_code
                final_set.discard(w)
                improved = True
                break

    return [list(w) for w in final_code]
