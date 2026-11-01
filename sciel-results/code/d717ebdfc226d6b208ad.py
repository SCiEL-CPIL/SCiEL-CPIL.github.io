import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Helper to convert index to tuple of digits
    def index_to_digits(idx):
        digits = [0] * n
        for k in range(n):
            digits[k] = idx % q
            idx //= q
        return digits

    # Precompute spheres as numpy arrays for faster slicing
    # spheres[i] is a numpy array of indices within radius of i
    spheres = []
    for i in range(total_words):
        base_digits = index_to_digits(i)
        sphere_set = set()
        for r in range(radius + 1):
            for pos_subset in itertools.combinations(range(n), r):
                if r == 0:
                    sphere_set.add(i)
                    continue
                for vals in itertools.product(range(q), repeat=r):
                    new_digits = base_digits[:]
                    for k, pos in enumerate(pos_subset):
                        new_digits[pos] = vals[k]
                    # Calculate index
                    idx = 0
                    for k in range(n - 1, -1, -1):
                        idx = idx * q + new_digits[k]
                    sphere_set.add(idx)
        spheres.append(np.array(list(sphere_set), dtype=np.int32))

    # State: in_code (bool array), cover_count (int array)
    in_code = np.zeros(total_words, dtype=bool)
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    def add_word(idx):
        if in_code[idx]:
            return
        in_code[idx] = True
        cover_count[spheres[idx]] += 1
        
    def remove_word(idx):
        if not in_code[idx]:
            return
        in_code[idx] = False
        cover_count[spheres[idx]] -= 1
        
    def greedy_repair():
        # While not all covered
        while np.any(cover_count < 1):
            # Find word covering most uncovered
            # uncovered_mask = cover_count < 1
            # For each i, count how many of spheres[i] are in uncovered_mask
            # This is slow if done naively.
            # Optimization: Only check candidates?
            # For N=2187, checking all is okay if vectorized.
            uncovered = cover_count < 1
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                if in_code[i]:
                    continue
                # Count how many words in sphere[i] are currently uncovered
                count = np.sum(uncovered[spheres[i]])
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break # Should not happen
            add_word(best_idx)

    def local_optimize():
        changed = True
        while changed:
            changed = False
            # Find redundant words
            # A word i is redundant if min(cover_count[spheres[i]]) > 1
            # We can check all in_code words
            for i in np.where(in_code)[0]:
                if np.min(cover_count[spheres[i]]) > 1:
                    remove_word(i)
                    changed = True
                    # Restart to check new redundancies
                    # Actually, we can just continue, but restart is safer for correctness
                    # Let's just break and restart the while loop
                    break

    best_size = float('inf')
    best_in_code = None
    
    num_runs = 50 # Adjust based on time
    start_time = time.time()
    
    for run in range(num_runs):
        if time.time() - start_time > 100: # Leave 20s buffer
            break
            
        in_code[:] = False
        cover_count[:] = 0
        
        # Strategy: Random Start + Repair + Optimize
        # Or Greedy Start + Optimize + Perturb
        
        if run == 0:
            # Pure Greedy
            greedy_repair()
        else:
            # Randomized Greedy or Random Start
            # Let's try Random Start with size ~1.5 * expected
            # Expected is ~186. Let's try 220.
            num_random = 220
            candidates = np.random.choice(total_words, num_random, replace=False)
            for c in candidates:
                add_word(c)
            # Repair
            greedy_repair()
            
        local_optimize()
        
        current_size = np.sum(in_code)
        if current_size < best_size:
            best_size = current_size
            best_in_code = in_code.copy()
            
        # Perturbation for next iteration (if we were doing ILS on the same state)
        # But here we restart each run.
        # To diversify, we can use different random seeds or strategies.
        
    # Extract codewords
    result_indices = np.where(best_in_code)[0]
    result = [index_to_digits(idx) for idx in result_indices]
    
    return result
