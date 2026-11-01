import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 113.0  # Safety margin for 120s limit
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Convert to numpy arrays for faster vectorized operations
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
    # Precompute ball membership matrix: ball_mem[i, j] = True if j is in ball(i)
    # This allows O(1) membership checks and fast vectorized operations
    ball_mem = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mem[i, ball_indices_np[i]] = True
    
    # Greedy construction with randomization
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            best_c = -1
            max_new = -1
            candidates = list(range(total))
            rng.shuffle(candidates)
            
            for c in candidates:
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code
    
    # Multiple greedy restarts to find good starting points
    best_initial = None
    best_initial_size = float('inf')
    
    init_code = greedy_build(seed=0)
    best_initial = set(init_code)
    best_initial_size = len(init_code)
    
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 40.0:
            break
        init_code = greedy_build(seed=seed)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = set(init_code)
    
    current_code = set(best_initial)
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1
        
    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns number removed."""
        removed = 0
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                ball = ball_indices_np[c]
                if np.min(cnt_arr[ball]) >= 2:
                    code_set.remove(c)
                    cnt_arr[ball] -= 1
                    removed += 1
                    changed = True
        return removed
    
    # Compute valid swaps using vectorized operations with ball_mem
    def get_valid_swaps(c, code_set, cnt_arr):
        """Return list of valid c_add for removing c using vectorized ball_mem."""
        ball_rem = ball_indices_np[c]
        
        # Words in ball_rem that would become uncovered if c is removed
        # These are words where cnt[x] == 1 (only covered by c)
        unique_mask = cnt_arr[ball_rem] == 1
        unique_words = ball_rem[unique_mask]
        
        if len(unique_words) == 0:
            # Any word not in code_set is valid
            return None
        
        # For a candidate c_add to be valid, it must cover all unique_words
        # i.e., for all x in unique_words, x must be in ball(c_add)
        # Using ball_mem: ball_mem[c_add, x] must be True for all x in unique_words
        
        # Vectorized check: for each candidate c_add, check if it covers all unique_words
        # We can use matrix multiplication or logical operations
        
        # Create a boolean matrix: covers_all[c_add] = all(ball_mem[c_add, unique_words])
        # This is equivalent to: sum(ball_mem[:, unique_words], axis=1) == len(unique_words)
        
        covers_count = np.sum(ball_mem[:, unique_words], axis=1)
        valid_mask = covers_count == len(unique_words)
        
        # Filter out words already in code_set
        code_mask = np.zeros(total, dtype=bool)
        for c_idx in code_set:
            code_mask[c_idx] = True
        
        valid_mask &= ~code_mask
        
        valid_list = np.where(valid_mask)[0].tolist()
        return valid_list
    
    # Local Search Loop with simulated annealing-inspired acceptance
    no_improve_counter = 0
    max_no_improve = 20
    temperature = 5.0  # Initial "temperature" for occasional worse moves
    min_temperature = 0.1
    
    while time.time() < deadline - 3.0:
        # 1. Delete Phase
        removed = run_delete_phase(current_code, cnt)
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1
        
        # 2. Swap Phase with smarter candidate selection
        swaps_attempted = 0
        max_swaps = 800
        while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
                
            # Pick random codeword to remove
            c = random.choice(list(current_code))
            
            # Get valid swaps
            valid_swaps = get_valid_swaps(c, current_code, cnt)
            
            if valid_swaps is None:
                # Any non-code word works; pick one that maximizes new coverage
                # of words with cnt==1 in ball(c)
                ball_rem = ball_indices_np[c]
                unique_mask = cnt[ball_rem] == 1
                unique_words = ball_rem[unique_mask]
                
                if len(unique_words) == 0:
                    # No unique words, pick random non-code word
                    c_prime = random.randint(0, total - 1)
                    attempts = 0
                    while c_prime in current_code and attempts < 100:
                        c_prime = random.randint(0, total - 1)
                        attempts += 1
                else:
                    # Pick from words that cover the most unique words
                    covers_count = np.sum(ball_mem[:, unique_words], axis=1)
                    code_mask = np.zeros(total, dtype=bool)
                    for c_idx in current_code:
                        code_mask[c_idx] = True
                    covers_count[code_mask] = -1
                    c_prime = int(np.argmax(covers_count))
                    
            elif len(valid_swaps) == 0:
                no_improve_counter += 1
                continue
            else:
                # Pick from valid swaps, preferring those that cover many unique words
                ball_rem = ball_indices_np[c]
                unique_mask = cnt[ball_rem] == 1
                unique_words = ball_rem[unique_mask]
                
                if len(unique_words) > 0:
                    covers_count = np.array([np.sum(ball_mem[v, unique_words]) for v in valid_swaps])
                    # Weighted random selection: prefer higher coverage
                    # Use softmax-like weighting
                    max_cov = np.max(covers_count)
                    weights = np.exp((covers_count - max_cov) / max(temperature, 0.01))
                    weights /= np.sum(weights)
                    c_prime = valid_swaps[random.choices(range(len(valid_swaps)), weights=weights)[0]]
                else:
                    c_prime = random.choice(valid_swaps)
            
            # Perform swap
            current_code.remove(c)
            current_code.add(c_prime)
            cnt[ball_indices_np[c]] -= 1
            cnt[ball_indices_np[c_prime]] += 1
            
            # After swap, run delete phase
            removed = run_delete_phase(current_code, cnt)
            if removed > 0:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            else:
                no_improve_counter += 1
            
            # Cool down temperature
            temperature = max(min_temperature, temperature * 0.995)
        
        # If no improvement for many iterations, try a restart
        if no_improve_counter > max_no_improve and time.time() < deadline - 15.0:
            # Try a new greedy initialization
            seed = int(time.time() * 1000) % 10000
            init_code = greedy_build(seed=seed)
            if len(init_code) < len(current_code) + 5:
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    cnt[ball_indices_np[c]] += 1
                no_improve_counter = 0
                temperature = 5.0  # Reset temperature on restart
                
        if best_size <= 160:
            break
    
    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
