import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Fallback for non-standard parameters
    if not (q == 3 and n == 6 and radius == 1):
        # Simple greedy construction
        # This might not be optimal but valid
        # For small n, brute force search for minimal code
        # For large n, product construction
        if n <= 4:
            # Brute force
            all_words = list(itertools.product(range(q), repeat=n))
            # Find minimal covering code
            # This is NP-hard, but n is small
            # Use greedy
            covered = set()
            code = []
            # Sort words by some heuristic? Or just random
            random.shuffle(all_words)
            for w in all_words:
                # Check if w is covered
                is_covered = False
                for c in code:
                    dist = sum(1 for i in range(n) if w[i] != c[i])
                    if dist <= radius:
                        is_covered = True
                        break
                if not is_covered:
                    code.append(w)
                    # Update covered set? 
                    # For simplicity, just re-check all words against code
            # Verify
            # This is O(N * 3^n * n). For n=4, 81*4 is small.
            return code
        else:
            # Product construction
            # Use tetracode for first 4 if q=3
            if q == 3:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                # General fallback: all zeros? No, must cover.
                # Just return a large set
                # Greedy
                all_words = list(itertools.product(range(q), repeat=n))
                random.shuffle(all_words)
                code = []
                for w in all_words:
                    is_covered = False
                    for c in code:
                        if sum(1 for i in range(n) if w[i] != c[i]) <= radius:
                            is_covered = True
                            break
                    if not is_covered:
                        code.append(w)
                return code

    # Specific case q=3, n=6, radius=1
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute coverage masks
    masks = np.zeros((total_words, total_words), dtype=np.int8)
    
    for i in range(total_words):
        w_i = idx_to_word(i)
        masks[i, i] = 1
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                w_neighbor = w_i.copy()
                w_neighbor[pos] = val
                j = word_to_idx(w_neighbor)
                masks[i, j] = 1

    # Initialize with the product code (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            init_words.append(w)
            
    init_indices = [word_to_idx(w) for w in init_words]
    
    # Initial coverage
    coverage = np.sum(masks[init_indices], axis=0)
    
    def minimize(code_indices, coverage):
        current_code = list(code_indices)
        current_cov = coverage.copy()
        
        while True:
            removed = False
            indices_to_check = current_code.copy()
            random.shuffle(indices_to_check)
            
            for idx in indices_to_check:
                if idx not in current_code:
                    continue
                new_cov = current_cov - masks[idx]
                if np.min(new_cov) >= 1:
                    current_code.remove(idx)
                    current_cov = new_cov
                    removed = True
                    break
            if not removed:
                break
        return current_code, current_cov

    code, coverage = minimize(init_indices, coverage)
    best_code = code.copy()
    best_cov = coverage.copy()
    best_size = len(code)
    
    while time.time() < deadline:
        if not code:
            break
        old_idx = random.choice(code)
        new_idx = random.randint(0, total_words - 1)
        
        if new_idx in code:
            continue
            
        new_cov = coverage - masks[old_idx] + masks[new_idx]
        
        if np.min(new_cov) < 1:
            continue
            
        new_code_list = [i for i in code if i != old_idx]
        new_code_list.append(new_idx)
        
        min_code, min_cov = minimize(new_code_list, new_cov)
        
        if len(min_code) < best_size:
            best_code = min_code
            best_cov = min_cov
            best_size = len(min_code)
            code = min_code
            coverage = min_cov
            
    result = [idx_to_word(i) for i in best_code]
    
    # Final validity check
    final_cov = np.sum(masks[best_code], axis=0)
    if np.min(final_cov) < 1:
        # Fallback
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] + list(tail) for a in range(3) for b in range(3) for tail in itertools.product(range(3), repeat=n - 4)]
        
    return result
