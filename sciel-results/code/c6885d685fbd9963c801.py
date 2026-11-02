import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute Hamming distance balls
    # ball_masks[i] is a bitmask where bit j is 1 if all_words[j] is within radius of all_words[i]
    ball_masks = [0] * total_words
    
    # Efficiently compute ball masks
    # For each word, find all words within radius
    for i, w in enumerate(all_words):
        mask = 0
        for j, v in enumerate(all_words):
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                mask |= (1 << j)
        ball_masks[i] = mask

    all_ones = (1 << total_words) - 1
    
    # Helper to calculate uncovered count for a set of indices
    # We maintain a coverage mask to make this fast
    # But for SA, we might just recalculate or track delta.
    # Recalculating is O(K) bitwise ORs. K~80. 80 ops. Very fast.
    
    def get_uncovered_count(code_indices):
        covered = 0
        for idx in code_indices:
            covered |= ball_masks[idx]
        return (all_ones ^ covered).bit_count()

    def get_cost(code_indices, penalty=100):
        return len(code_indices) + penalty * get_uncovered_count(code_indices)

    # 1. Generate initial solution using greedy cover
    def greedy_cover():
        uncovered = all_ones
        code_indices = []
        while uncovered != 0:
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                new_points = uncovered & ball_masks[i]
                count = new_points.bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            code_indices.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return code_indices

    current_code = greedy_cover()
    current_cost = get_cost(current_code)
    best_code = current_code[:]
    best_cost = current_cost
    best_uncovered = get_uncovered_count(best_code)

    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute list of all indices for random selection
    all_indices = list(range(total_words))
    
    while time.time() < deadline - 5.0 and T > T_min:
        # Propose a move
        move_type = random.random()
        
        new_code = current_code[:]
        
        if move_type < 0.4:
            # Add a random word
            new_code.append(random.choice(all_indices))
        elif move_type < 0.8 and len(current_code) > 0:
            # Remove a random word
            idx_to_remove = random.randint(0, len(current_code) - 1)
            new_code.pop(idx_to_remove)
        else:
            # Swap or Replace
            if len(current_code) > 0:
                if random.random() < 0.5:
                    # Swap 1 with 1
                    i = random.randint(0, len(current_code) - 1)
                    new_code[i] = random.choice(all_indices)
                else:
                    # Replace 2 with 2 (if size >= 2)
                    if len(current_code) >= 2:
                        i1 = random.randint(0, len(current_code) - 1)
                        i2 = random.randint(0, len(current_code) - 1)
                        if i1 != i2:
                            new_code[i1] = random.choice(all_indices)
                            new_code[i2] = random.choice(all_indices)
                    else:
                        # Fallback to swap
                        i = random.randint(0, len(current_code) - 1)
                        new_code[i] = random.choice(all_indices)
            
        # Ensure distinctness? The problem says codewords must be distinct.
        # If we add a duplicate, it doesn't help coverage but increases size.
        # SA will naturally reject it if it doesn't improve uncovered count enough.
        # But to be safe, we can check uniqueness.
        # However, checking uniqueness in a list is O(K).
        # We can just allow it and rely on the cost function (size increases).
        
        new_cost = get_cost(new_code)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
            current_code = new_code
            current_cost = new_cost
            
            # Check if valid and better
            unc = get_uncovered_count(current_code)
            if unc == 0:
                if len(current_code) < len(best_code):
                    best_code = current_code[:]
                    best_cost = current_cost
                    best_uncovered = 0
        
        T *= cooling_rate
        
        # Occasionally try to minimize the current valid code if we have a valid one
        # But SA itself does this.
        
    # Final check: if best_code is not valid, use current_code if it is better, or try to fix
    # Actually, if best_uncovered > 0, we should return the one with min uncovered, then min size?
    # The task requires validity. If we can't find a valid one, we return the best effort?
    # The evaluator says "Valid when...". If invalid, score 0.
    # So we MUST return a valid code.
    # Greedy always produces a valid code.
    # So best_code from Greedy is valid.
    # SA might find a better valid code.
    
    # If best_code is not valid (shouldn't happen if we start with valid), 
    # we should ensure we return a valid one.
    if get_uncovered_count(best_code) != 0:
        # Fallback to the first valid one found (Greedy)
        # Greedy is always valid.
        best_code = greedy_cover()

    # Final minimization on the best valid code
    def minimize_code(code_indices):
        changed = True
        while changed:
            changed = False
            for i in range(len(code_indices)):
                new_code = code_indices[:i] + code_indices[i+1:]
                if get_uncovered_count(new_code) == 0:
                    code_indices = new_code
                    changed = True
                    break
        return code_indices

    best_code = minimize_code(best_code)
    
    code_words = [list(all_words[idx]) for idx in best_code]
    return code_words
