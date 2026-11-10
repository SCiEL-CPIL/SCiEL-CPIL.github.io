import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # This task is specifically for q=3, n=7, radius=1.
    # The contract says the evaluator passes q, n, radius.
    # We assume the standard case q=3, n=7, radius=1 based on the task description.
    # If parameters differ, we might need a fallback, but the task is "Football pool problem for 7 matches".
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to simple construction for other cases (though unlikely to be tested)
        # Direct sum of tetracode if possible, else random
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Brute force small cases
            import itertools
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    total_words = q ** n  # 2187
    
    # Helper to convert tuple of digits to integer index
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute all words and their indices
    all_words = []
    for i in range(total_words):
        all_words.append(idx_to_word(i))
        
    # Precompute spheres (list of indices covered by each word)
    # sphere[i] is a set of indices j such that dist(i, j) <= radius
    spheres = [set() for _ in range(total_words)]
    
    # Efficient sphere computation
    # For each word, generate all neighbors within distance 1
    for i in range(total_words):
        w = all_words[i]
        spheres[i].add(i)
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    new_word = w.copy()
                    new_word[pos] = new_val
                    j = word_to_idx(new_word)
                    spheres[i].add(j)
                    
    # Represent spheres as bitsets for fast intersection
    # Python integers can hold 2187 bits easily
    sphere_bits = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in spheres[i]:
            mask |= (1 << j)
        sphere_bits[i] = mask
        
    best_code = None
    best_size = float('inf')
    
    # Number of greedy runs
    num_runs = 50
    
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        # Greedy Set Cover
        while uncovered:
            best_gain = -1
            best_c = -1
            # Randomize tie-breaking to explore different solutions
            # We iterate through all candidates. To randomize, we can shuffle the order of checks or add small noise.
            # Shuffling 2187 items 186 times is expensive.
            # Instead, we just pick the first max. To vary, we can randomize the initial order of candidates.
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for c in candidates:
                # Calculate gain: number of bits in (sphere_bits[c] & uncovered)
                # Using bin(x).count('1') is fast enough for 2187 bits
                gain = bin(sphere_bits[c] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                # Optional: Early break if gain is max possible? Max is 15.
                if best_gain == 15:
                    break
                    
            if best_c == -1:
                break
                
            code_indices.append(best_c)
            uncovered &= ~sphere_bits[best_c]
            
        # Pruning Phase
        # Maintain coverage count
        # count[j] = number of codewords in code_indices that cover j
        count = [0] * total_words
        for c in code_indices:
            for j in spheres[c]:
                count[j] += 1
                
        # Try to remove codewords
        # We iterate through the code. If removing c keeps all counts > 0, remove it.
        # We do this in a random order to break symmetry.
        to_check = list(code_indices)
        random.shuffle(to_check)
        
        current_code = set(code_indices)
        
        for c in to_check:
            if time.time() > deadline:
                break
            if c not in current_code:
                continue
                
            # Check if removing c is safe
            safe = True
            for j in spheres[c]:
                if count[j] <= 1:
                    safe = False
                    break
                    
            if safe:
                # Remove c
                current_code.remove(c)
                for j in spheres[c]:
                    count[j] -= 1
        
        # Final check: is the code valid?
        # The pruning logic ensures validity if it started valid.
        # But let's verify size.
        final_code_list = list(current_code)
        
        if len(final_code_list) < best_size:
            best_size = len(final_code_list)
            best_code = final_code_list
            
            # If we reached the theoretical lower bound or close to it, we can stop
            if best_size <= 186:
                break

    if best_code is None:
        # Fallback to seed solution if greedy failed or timed out immediately
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to word lists
    result = [all_words[i] for i in best_code]
    return result
