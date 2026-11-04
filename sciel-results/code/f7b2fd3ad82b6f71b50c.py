import numpy as np
import random
import itertools
import time

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple greedy for other parameters, though task is specific
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    neighbors[i].append(word_to_idx[new_word])
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = sum(1 for nb in neighbors[c] if nb in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            for nb in neighbors[best_c]:
                if nb in uncovered:
                    uncovered.remove(nb)
        
        return [list(all_words[i]) for i in solution]

    # Specific optimization for q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words) # 2187
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (distance <= 1)
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                new_word = list(w)
                new_word[pos] = val
                new_word = tuple(new_word)
                neighbors[i].append(word_to_idx[new_word])
        
    # 1. Initial Greedy Solution
    # Fast greedy using sets
    uncovered = set(range(N))
    current_code = []
    while uncovered:
        best_c = -1
        best_gain = -1
        # Randomize tie-breaking slightly to vary starting point? 
        # For reproducibility, we use deterministic, but random seed is set later.
        # To get a good start, we can use a simple greedy.
        for c in range(N):
            gain = 0
            for nb in neighbors[c]:
                if nb in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        current_code.append(best_c)
        for nb in neighbors[best_c]:
            if nb in uncovered:
                uncovered.remove(nb)
                
    # 2. Local Search: Remove & Repair
    # Maintain cover_count for fast updates
    cover_count = [0] * N
    for c in current_code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    best_code = list(current_code)
    best_size = len(best_code)
    
    start_time = time.time()
    timeout = 115.0
    random.seed(42)
    
    # We will perform "Remove and Repair" moves.
    # 1. Pick a random codeword to remove.
    # 2. Decrement counts.
    # 3. Find uncovered words.
    # 4. If none, accept.
    # 5. If some, find best codeword to cover them (greedy repair on uncovered set).
    # 6. If new size < best_size, update best.
    
    while time.time() - start_time < timeout:
        if not current_code:
            break
            
        # Pick a random codeword to remove
        idx_to_remove = random.randint(0, len(current_code) - 1)
        c_to_remove = current_code[idx_to_remove]
        
        # Tentatively remove
        uncovered = []
        for nb in neighbors[c_to_remove]:
            cover_count[nb] -= 1
            if cover_count[nb] == 0:
                uncovered.append(nb)
                
        current_code.pop(idx_to_remove)
        
        if not uncovered:
            # Successful removal
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            continue
            
        # Need to repair
        # Greedy repair on the uncovered set
        remaining_uncovered = set(uncovered)
        
        # We might need to add multiple codewords
        repair_count = 0
        while remaining_uncovered:
            best_c_new = -1
            best_gain = -1
            for c in range(N):
                gain = 0
                for nb in neighbors[c]:
                    if nb in remaining_uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c_new = c
                    
            if best_gain == 0:
                # Should not happen
                break
                
            current_code.append(best_c_new)
            repair_count += 1
            
            for nb in neighbors[best_c_new]:
                cover_count[nb] += 1
                if nb in remaining_uncovered:
                    remaining_uncovered.remove(nb)
                    
        # Evaluate move
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
        else:
            # Revert if it didn't improve size, to focus on minimization
            # This is a "Best Improvement" strategy.
            # To allow exploration, we could accept same-size moves with probability p.
            # But strict minimization is safer for the score.
            # Let's revert.
            for _ in range(repair_count):
                c_added = current_code.pop()
                for nb in neighbors[c_added]:
                    cover_count[nb] -= 1
            current_code.append(c_to_remove)
            for nb in neighbors[c_to_remove]:
                cover_count[nb] += 1

    return [list(all_words[i]) for i in best_code]
