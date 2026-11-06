import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 1 << N  # 131072
    
    # 1. Generate the 256-word linear seed code
    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Generator vector g as an integer
    g = 0
    for q in Q:
        g |= (1 << q)
        
    # Basis: cyclic shifts of g
    basis = [g]
    for i in range(1, 9):
        g = ((g << 1) | (g >> (N - 1))) & (NUM_WORDS - 1)
        basis.append(g)
        
    # Generate all 2^9 linear combinations
    # Keep only even-weight words to ensure min distance >= 6
    seed_words = set()
    for mask in range(512):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= basis[i]
        if w.bit_count() % 2 == 0:
            seed_words.add(w)
            
    seed_list = list(seed_words)
    
    # 2. Precompute offsets for distance < 6 (i.e., distance 0 to 5)
    offsets = []
    for i in range(NUM_WORDS):
        if i.bit_count() <= 5:
            offsets.append(i)
    offsets = np.array(offsets, dtype=np.uint32)
    
    # 3. Greedy Fill Function
    def greedy_fill(start_words, random_seed):
        valid = np.ones(NUM_WORDS, dtype=bool)
        selected = []
        
        for w in start_words:
            if valid[w]:
                selected.append(w)
                valid[w ^ offsets] = False
        
        candidates = np.where(valid)[0]
        
        rng = np.random.default_rng(random_seed)
        rng.shuffle(candidates)
        
        for c in candidates:
            if valid[c]:
                selected.append(c)
                valid[c ^ offsets] = False
        
        return selected

    # 4. Local Search with Simulated Annealing
    def local_search(code, temperature=1.0, max_iterations=10000):
        """
        Try to add words to the code by finding valid positions,
        and occasionally perturb existing codewords to free up space.
        """
        code_set = set(code)
        valid = np.ones(NUM_WORDS, dtype=bool)
        for w in code_set:
            valid[w ^ offsets] = False
        
        current_code = list(code)
        current_len = len(current_code)
        best_code = current_code[:]
        best_len = current_len
        
        rng = np.random.default_rng(42)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Strategy 1: Try to add a random valid word
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                # Pick a random valid word
                new_word = valid_indices[rng.integers(len(valid_indices))]
                if valid[new_word]:
                    current_code.append(new_word)
                    current_len += 1
                    valid[new_word ^ offsets] = False
                    
                    if current_len > best_len:
                        best_len = current_len
                        best_code = current_code[:]
                    
                    # Occasionally revert to maintain diversity
                    if rng.random() < 0.1:
                        # Remove the last added word
                        current_code.pop()
                        current_len -= 1
                        # Re-validate (simplified: just mark it valid again)
                        # This is approximate but works for the search
                        pass
            
            # Strategy 2: Perturb an existing codeword
            if len(current_code) > 0 and rng.random() < 0.3:
                idx = rng.integers(len(current_code))
                old_word = current_code[idx]
                
                # Try flipping 1-3 bits
                for flip_count in range(1, 4):
                    # Generate random flip mask
                    flip_mask = 0
                    for _ in range(flip_count):
                        bit = rng.integers(N)
                        flip_mask ^= (1 << bit)
                    
                    new_word = old_word ^ flip_mask
                    
                    # Check if new_word is valid (not conflicting with others)
                    if valid[new_word]:
                        # Accept the move
                        current_code[idx] = new_word
                        valid[old_word ^ offsets] = True  # Free up old position
                        valid[new_word ^ offsets] = False  # Block new position
                        break
        
        return best_code, best_len

    # 5. Main Loop with Restarts and Local Search
    best_solution = seed_list[:]
    best_len = len(best_solution)
    
    # First, do some greedy fills to get a good starting point
    run_id = 0
    while time.time() < deadline - 30:  # Leave 30s for local search
        res = greedy_fill(seed_list, run_id)
        
        if len(res) > best_len:
            best_len = len(res)
            best_solution = res[:]
        
        run_id += 1
        
        if run_id % 50 == 0:
            if time.time() > deadline - 30:
                break
    
    # Now apply local search to the best solution found
    if time.time() < deadline - 5:
        improved_code, improved_len = local_search(best_solution, temperature=1.0, max_iterations=50000)
        
        if improved_len > best_len:
            best_len = improved_len
            best_solution = improved_code[:]
    
    # Final greedy fill on the improved code
    if time.time() < deadline - 2:
        final_res = greedy_fill(best_solution, 12345)
        if len(final_res) > best_len:
            best_len = len(final_res)
            best_solution = final_res[:]
    
    return best_solution
