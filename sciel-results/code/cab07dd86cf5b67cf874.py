import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # 1. Generate initial seed: Direct sum of tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Convert to base-3 integers for efficiency
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * q + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n):
            word[i] = val % q
            val //= q
        return word

    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_int(word))
    
    code = set(initial_codewords)
    
    # 2. Precompute balls for all q^n words using numpy for speed
    total_words = q ** n
    all_words = np.arange(total_words)
    
    # Precompute digit values for all positions
    # digits[w, i] = digit at position i of word w
    digits = np.zeros((total_words, n), dtype=np.int32)
    for i in range(n):
        div = q ** (n - 1 - i)
        digits[:, i] = (all_words // div) % q
    
    # Precompute balls: for each word, store its neighbors as indices
    # Ball of w with radius 1: w itself + all words differing in exactly one position
    # For position i, changing digit to d (d != current) gives a neighbor
    balls = []
    for w in range(total_words):
        b = [w]
        w_list = int_to_list(w, n)
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    b.append(list_to_int(new_word))
        balls.append(b)
    
    # Convert balls to numpy arrays for faster operations
    balls_np = [np.array(b, dtype=np.int32) for b in balls]
    
    # 3. Greedy Removal - batch removal of all non-essential codewords
    def batch_remove():
        nonlocal code
        if time.time() > deadline:
            return False
        
        # Calculate coverage counts and unique coverers using numpy
        coverage = np.zeros(total_words, dtype=np.int32)
        unique_coverer = np.full(total_words, -1, dtype=np.int32)
        
        for c in code:
            for w in balls_np[c]:
                if coverage[w] == 0:
                    unique_coverer[w] = c
                elif coverage[w] == 1:
                    unique_coverer[w] = -1
                coverage[w] += 1
        
        # Find essential codewords
        essential = set()
        uc_arr = unique_coverer
        for w in range(total_words):
            if uc_arr[w] >= 0:
                essential.add(uc_arr[w])
        
        removable = code - essential
        
        if not removable:
            return False
            
        # Remove all removable codewords at once
        code -= removable
        return True
    
    # 4. Local Search (1-swap): try to replace a codeword with a better one
    def try_swap():
        nonlocal code
        if time.time() > deadline:
            return False
            
        # Calculate coverage
        coverage = np.zeros(total_words, dtype=np.int32)
        unique_coverer = np.full(total_words, -1, dtype=np.int32)
        
        for c in code:
            for w in balls_np[c]:
                if coverage[w] == 0:
                    unique_coverer[w] = c
                elif coverage[w] == 1:
                    unique_coverer[w] = -1
                coverage[w] += 1
        
        # Group words by unique coverer
        unique_groups = {}
        for w in range(total_words):
            uc = unique_coverer[w]
            if uc >= 0:
                if uc not in unique_groups:
                    unique_groups[uc] = []
                unique_groups[uc].append(w)
        
        # Sort by size of unique group to find easy swaps first
        candidates = sorted(unique_groups.items(), key=lambda x: len(x[1]))
        
        for c, U_c in candidates:
            if len(U_c) == 0:
                continue
            
            # Try to find w_new that covers U_c
            # The new codeword must be within distance 1 of all words in U_c
            # This means w_new must be in the intersection of balls[U_c]
            # Start with the ball of the first word, intersect with others
            
            candidate_set = set(balls_np[U_c[0]].tolist())
            for w in U_c[1:]:
                candidate_set &= set(balls_np[w].tolist())
                if not candidate_set:
                    break
            
            # Filter to words not in code
            candidate_set -= code
            
            # Check each candidate
            for w_new in candidate_set:
                # Verify it covers all of U_c
                if all(w in set(balls_np[w_new].tolist()) for w in U_c):
                    # Perform swap: remove c, add w_new
                    code.remove(c)
                    code.add(w_new)
                    return True
                    
        return False
    
    # 5. Repair: if any word is uncovered, add a codeword to cover it
    def repair():
        nonlocal code
        if time.time() > deadline:
            return False
        
        # Find uncovered words
        covered = np.zeros(total_words, dtype=bool)
        for c in code:
            covered[balls_np[c]] = True
        
        uncovered = np.where(~covered)[0]
        
        if len(uncovered) == 0:
            return False
        
        # Add a codeword that covers the most uncovered words
        best_c = None
        best_count = 0
        for w in range(total_words):
            if w in code:
                continue
            count = np.sum(covered[balls_np[w]] == False)
            if count > best_count:
                best_count = count
                best_c = w
        
        if best_c is not None and best_count > 0:
            code.add(best_c)
            return True
        
        return False
    
    # Main Loop
    while time.time() < deadline:
        made_progress = False
        
        # Try batch removal
        if batch_remove():
            made_progress = True
            continue
        
        # Try swap - attempt multiple times in a row
        swap_attempts = 0
        max_swap_attempts = 5
        while swap_attempts < max_swap_attempts and time.time() < deadline:
            if try_swap():
                made_progress = True
                swap_attempts += 1
            else:
                break
        
        if made_progress:
            continue
        
        # Try repair (should not be needed if we maintain coverage, but just in case)
        if repair():
            made_progress = True
            continue
        
        # If stuck, try random perturbation: remove a random codeword and repair
        if len(code) > 0 and random.random() < 0.15:
            c_to_remove = random.choice(list(code))
            code.remove(c_to_remove)
            # Repair will fix coverage
            continue
        
        # No progress, break
        break
    
    # Final batch removal pass
    while time.time() < deadline:
        if not batch_remove():
            break
    
    # Convert back to list of lists
    result = [int_to_list(c, n) for c in code]
    return result

# EVOLVE-BLOCK-END
