import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

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
    # We also need the transpose for "who covers j"
    ball_mem = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mem[i, ball_indices_np[i]] = True
    ball_mem_T = ball_mem.T

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

    # First greedy is deterministic
    init_code = greedy_build(seed=0)
    best_initial = init_code
    best_initial_size = len(init_code)

    # Try a few more with different seeds, but limit time
    for seed in [1, 2, 3]:
        if time.time() > deadline - 40.0:
            break
        init_code = greedy_build(seed=seed)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = init_code

    # Initialize state
    current_code = list(best_initial)
    in_code = np.zeros(total, dtype=bool)
    for c in current_code:
        in_code[c] = True
        
    best_code = list(current_code)
    best_size = len(current_code)

    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1

    def run_delete_phase():
        """Removes redundant codewords. Returns number removed."""
        removed = 0
        changed = True
        # Iterate over a copy of the list
        i = 0
        while i < len(current_code):
            c = current_code[i]
            ball = ball_indices_np[c]
            if np.min(cnt[ball]) >= 2:
                # Remove c
                current_code.pop(i)
                in_code[c] = False
                cnt[ball] -= 1
                removed += 1
                # Do not increment i, check the new element at i
            else:
                i += 1
        return removed

    def get_valid_add_candidates(c_remove):
        """Find valid c_add candidates for removing c_remove."""
        ball_rem = ball_indices_np[c_remove]
        # Find words in ball_rem that are only covered by c_remove
        unique_mask = cnt[ball_rem] == 1
        unique_words = ball_rem[unique_mask]
        
        if len(unique_words) == 0:
            # Any word not in code is valid
            # We return a mask of all non-code words
            valid_mask = ~in_code
            return valid_mask
        
        # Intersection of balls of all unique words
        # valid_mask[c] is True if c covers all unique_words
        # ball_mem_T[u, c] is True if c covers u
        valid_mask = np.ones(total, dtype=bool)
        valid_mask[in_code] = False # Cannot add existing code
        
        for u in unique_words:
            valid_mask &= ball_mem_T[u]
            if not np.any(valid_mask):
                return valid_mask # Empty
        
        return valid_mask

    # Local Search Loop
    no_improve_counter = 0
    max_no_improve = 20
    temperature = 1.0 # Not strictly needed if we only accept improvements, but good for neutral moves
    
    while time.time() < deadline - 5.0:
        # 1. Delete Phase
        removed = run_delete_phase()
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1
        
        # 2. Swap Phase
        swaps_attempted = 0
        max_swaps = 2000 # Increased budget
        while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
            
            # Pick random codeword to remove
            idx = random.randint(0, len(current_code) - 1)
            c_remove = current_code[idx]
            
            # Get valid add candidates
            valid_mask = get_valid_add_candidates(c_remove)
            
            if not np.any(valid_mask):
                no_improve_counter += 1
                continue
            
            # Pick a random candidate from valid_mask
            candidates = np.where(valid_mask)[0]
            c_add = candidates[random.randint(0, len(candidates) - 1)]
            
            # Perform swap
            # Remove c_remove
            current_code[idx] = c_add
            in_code[c_remove] = False
            in_code[c_add] = True
            cnt[ball_indices_np[c_remove]] -= 1
            cnt[ball_indices_np[c_add]] += 1
            
            # Run delete phase to see if we can shrink
            removed = run_delete_phase()
            
            if removed > 0:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    no_improve_counter = 0
                else:
                    # Neutral move (size didn't change, but we might have moved to a better local minimum)
                    # Accept it to explore
                    no_improve_counter += 1
            else:
                # No improvement, and no neutral benefit? 
                # Actually, if removed==0, the size is the same. 
                # We should probably revert if it doesn't help, OR keep it if we want to explore.
                # Let's revert if it doesn't lead to immediate deletion, to avoid random walk.
                # But random walk can help. Let's accept with small probability.
                if random.random() < 0.1:
                    no_improve_counter += 1
                else:
                    # Revert
                    current_code[idx] = c_remove
                    in_code[c_remove] = True
                    in_code[c_add] = False
                    cnt[ball_indices_np[c_remove]] += 1
                    cnt[ball_indices_np[c_add]] -= 1
                    no_improve_counter += 1

        # If no improvement for many iterations, try a restart
        if no_improve_counter > max_no_improve and time.time() < deadline - 20.0:
            # Try a new greedy initialization
            seed = int(time.time() * 1000) % 10000
            init_code = greedy_build(seed=seed)
            if len(init_code) < len(current_code) + 5:  # only if competitive
                current_code = list(init_code)
                in_code = np.zeros(total, dtype=bool)
                for c in current_code:
                    in_code[c] = True
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    cnt[ball_indices_np[c]] += 1
                no_improve_counter = 0
                
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
