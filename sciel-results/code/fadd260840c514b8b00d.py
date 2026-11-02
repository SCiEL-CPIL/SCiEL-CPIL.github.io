import numpy as np
import itertools
import random
import math
import time

def construct(q, n, radius):
    total_words = q ** n
    start_time = time.time()
    TIME_BUDGET = 115
    
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
            
    words = [to_word(i) for i in range(total_words)]
    
    # Precompute balls
    ball_masks = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        word = words[i]
        ball_masks[i, i] = True
        for pos in range(n):
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (word[pos] + delta) % q
                j = to_int(new_word)
                ball_masks[i, j] = True

    def greedy_init():
        uncovered_mask = np.ones(total_words, dtype=np.int8)
        code = set()
        while np.any(uncovered_mask):
            counts = ball_masks @ uncovered_mask
            for c in code:
                counts[c] = -1
            best_c = np.argmax(counts)
            if counts[best_c] <= 0:
                break
            code.add(best_c)
            uncovered_mask[ball_masks[best_c, :]] = 0
        return code

    def simulate_annealing(init_code, time_limit):
        code = set(init_code)
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cover_count[ball_masks[c, :]] += 1
            
        uncovered_count = np.sum(cover_count == 0)
        current_size = len(code)
        current_energy = current_size + 1000 * uncovered_count
        
        best_code = set(code)
        best_energy = current_energy
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        in_code = np.zeros(total_words, dtype=bool)
        for c in code:
            in_code[c] = True
            
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
                
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove
                if not code: continue
                c_out = random.choice(list(code))
                
                # Calculate change
                # Words that become uncovered: those with count 1 in ball(c_out)
                newly_uncovered = ball_masks[c_out, :] & (cover_count == 1)
                delta_uncovered = np.sum(newly_uncovered)
                
                # Apply
                code.remove(c_out)
                cover_count[ball_masks[c_out, :]] -= 1
                in_code[c_out] = False
                uncovered_count += delta_uncovered
                new_size = len(code)
                new_energy = new_size + 1000 * uncovered_count
                
                if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / T):
                    current_energy = new_energy
                    current_size = new_size
                    if uncovered_count == 0 and new_energy < best_energy:
                        best_energy = new_energy
                        best_code = set(code)
                else:
                    # Revert
                    code.add(c_out)
                    cover_count[ball_masks[c_out, :]] += 1
                    in_code[c_out] = True
                    uncovered_count -= delta_uncovered
                    
            elif move_type < 0.7:
                # Add
                c_in = random.randint(0, total_words - 1)
                if in_code[c_in]:
                    continue
                
                # Calculate change
                # Words that become covered: those with count 0 in ball(c_in)
                newly_covered = ball_masks[c_in, :] & (cover_count == 0)
                delta_uncovered = -np.sum(newly_covered)
                
                # Apply
                code.add(c_in)
                cover_count[ball_masks[c_in, :]] += 1
                in_code[c_in] = True
                uncovered_count += delta_uncovered
                new_size = len(code)
                new_energy = new_size + 1000 * uncovered_count
                
                if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / T):
                    current_energy = new_energy
                    current_size = new_size
                    if uncovered_count == 0 and new_energy < best_energy:
                        best_energy = new_energy
                        best_code = set(code)
                else:
                    # Revert
                    code.remove(c_in)
                    cover_count[ball_masks[c_in, :]] -= 1
                    in_code[c_in] = False
                    uncovered_count -= delta_uncovered
                    
            else:
                # Swap
                if not code: continue
                c_out = random.choice(list(code))
                c_in = random.randint(0, total_words - 1)
                if in_code[c_in] and c_in != c_out:
                    continue
                if c_in == c_out:
                    continue
                
                # Calculate change for Remove c_out
                newly_uncovered_out = ball_masks[c_out, :] & (cover_count == 1)
                delta_uncovered_out = np.sum(newly_uncovered_out)
                
                # Calculate change for Add c_in
                # Note: cover_count is updated after remove?
                # We should calculate based on current state, then apply.
                # But Add and Remove interact.
                # Let's just apply and then revert if needed.
                # Or calculate net change.
                
                # Net change in cover_count: -ball[c_out] + ball[c_in]
                # Words affected: Union(ball[c_out], ball[c_in])
                
                # Simpler: Apply Remove, calculate, Apply Add, calculate, then decide.
                # But we need to revert both if rejected.
                
                # Apply Remove
                code.remove(c_out)
                cover_count[ball_masks[c_out, :]] -= 1
                in_code[c_out] = False
                uncovered_count += delta_uncovered_out
                
                # Apply Add
                # Newly covered by c_in given the NEW cover_count (after remove)
                newly_covered_in = ball_masks[c_in, :] & (cover_count == 0)
                delta_uncovered_in = -np.sum(newly_covered_in)
                
                code.add(c_in)
                cover_count[ball_masks[c_in, :]] += 1
                in_code[c_in] = True
                uncovered_count += delta_uncovered_in
                
                new_size = len(code)
                new_energy = new_size + 1000 * uncovered_count
                
                if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / T):
                    current_energy = new_energy
                    current_size = new_size
                    if uncovered_count == 0 and new_energy < best_energy:
                        best_energy = new_energy
                        best_code = set(code)
                else:
                    # Revert Add
                    code.remove(c_in)
                    cover_count[ball_masks[c_in, :]] -= 1
                    in_code[c_in] = False
                    uncovered_count -= delta_uncovered_in
                    
                    # Revert Remove
                    code.add(c_out)
                    cover_count[ball_masks[c_out, :]] += 1
                    in_code[c_out] = True
                    uncovered_count -= delta_uncovered_out
                    
            T *= cooling
            
        return best_code

    best_overall = None
    best_overall_size = float('inf')
    
    init_code = greedy_init()
    
    # Quick polish
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in init_code:
        cover_count[ball_masks[c, :]] += 1
        
    changed = True
    while changed:
        changed = False
        for c in list(init_code):
            if np.min(cover_count[ball_masks[c, :]]) > 1:
                init_code.remove(c)
                cover_count[ball_masks[c, :]] -= 1
                changed = True
                
    if len(init_code) < best_overall_size:
        best_overall = set(init_code)
        best_overall_size = len(init_code)
        
    while time.time() - start_time < TIME_BUDGET - 5:
        perturbed_code = set(init_code)
        for _ in range(5):
            if len(perturbed_code) < 2: break
            c_out = random.choice(list(perturbed_code))
            c_in = random.randint(0, total_words - 1)
            perturbed_code.remove(c_out)
            perturbed_code.add(c_in)
            
        sa_result = simulate_annealing(perturbed_code, time_limit=time.time() - start_time + 10)
        
        if len(sa_result) < best_overall_size:
            best_overall = sa_result
            best_overall_size = len(sa_result)
            init_code = sa_result
            
    if not best_overall:
        best_overall = init_code
        
    result = [to_word(c) for c in best_overall]
    return result
