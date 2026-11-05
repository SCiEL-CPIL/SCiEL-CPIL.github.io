import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer for safety
    
    # 1. Generate all words
    all_words = []
    for w in itertools.product(range(q), repeat=n):
        all_words.append(w)
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # 2. Precompute neighbors (Hamming distance <= radius)
    neighbors = []
    for i in range(N):
        w_i = all_words[i]
        nbrs = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w = list(w_i)
                new_w[pos] = val
                nbrs.append(word_to_idx[tuple(new_w)])
        neighbors.append(nbrs)
        
    # 3. Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    cover_count = np.zeros(N, dtype=int)
    code = []
    
    # Precompute list of indices for faster iteration
    indices = list(range(N))
    
    while np.any(uncovered):
        if time.time() > deadline - 20.0: # Reserve time for local search
            break
        best_i = -1
        best_count = -1
        # Shuffle to avoid deterministic bias in greedy
        random.shuffle(indices)
        for i in indices:
            # If i is already covered, it might still be a good candidate?
            # No, we want to cover uncovered points.
            # But a codeword covers itself and neighbors.
            # So we count how many UNCOVERED points are in its ball.
            cnt = 0
            for j in neighbors[i]:
                if uncovered[j]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_i = i
        if best_i == -1:
            break
        # Add best_i
        code.append(best_i)
        for j in neighbors[best_i]:
            if uncovered[j]:
                uncovered[j] = False
            cover_count[j] += 1
            
    # 4. Local Search
    best_code = code.copy()
    best_size = len(code)
    
    def reduce_code():
        nonlocal best_size, best_code
        changed = True
        while changed:
            changed = False
            # Shuffle to randomize removal order
            current_code_copy = code.copy()
            random.shuffle(current_code_copy)
            for idx in current_code_copy:
                if idx not in code:
                    continue
                # Check if removable
                removable = True
                for j in neighbors[idx]:
                    if cover_count[j] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove
                    code.remove(idx)
                    for j in neighbors[idx]:
                        cover_count[j] -= 1
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                    # Restart the loop to check for new removabilities
                    break 

    # Initial reduction
    reduce_code()
    
    T = 1.0
    cooling_rate = 0.999
    
    # Local Search Loop
    while time.time() < deadline:
        if not code:
            break
            
        # Pick a random codeword to potentially swap
        i = random.choice(code)
        
        # Try removing i
        safe = True
        for j in neighbors[i]:
            if cover_count[j] <= 1:
                safe = False
                break
        
        if safe:
            # Temporarily remove i
            code.remove(i)
            for j in neighbors[i]:
                cover_count[j] -= 1
            
            # Pick a random new codeword
            j_new = random.randint(0, N - 1)
            
            # Add j_new
            code.append(j_new)
            for j in neighbors[j_new]:
                cover_count[j] += 1
            
            # Try to reduce
            reduce_code()
            
            # If no improvement, revert to explore other options
            if len(code) >= best_size:
                # Revert add j_new
                if j_new in code:
                    code.remove(j_new)
                for j in neighbors[j_new]:
                    cover_count[j] -= 1
                # Revert remove i
                if i not in code:
                    code.append(i)
                for j in neighbors[i]:
                    cover_count[j] += 1
            
            T = max(T * cooling_rate, 0.1)
        else:
            # Cannot remove i directly. 
            # We might try to add a new codeword first? 
            # This would increase size, which we want to avoid.
            # So we just skip.
            pass

    # Return the best code found
    return [list(all_words[i]) for i in best_code]
