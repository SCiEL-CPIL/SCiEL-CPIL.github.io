import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    # q=3, n=6, radius=1
    # Total words = 3**6 = 729
    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute neighbors for each word (Hamming distance <= 1)
    # A word is a tuple of n ints
    words = []
    for i in range(total_words):
        # Convert index to base-q tuple
        w = [0] * n
        temp = i
        for j in range(n):
            w[j] = temp % q
            temp //= q
        words.append(tuple(w))
        
    # Map index to word and word to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
    for i in range(total_words):
        w = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors.append(neigh)
    
    # --- Initial Code Construction ---
    # We start with a valid covering code. 
    # A simple valid code is all words where the sum of coordinates mod q is 0? 
    # No, that's a code of size q^(n-1) = 3^5 = 243. 
    # The tetracode direct sum approach from the original code gives 81 words.
    # Let's stick with the tetracode-based initialization as it's a good starting point.
    
    tetracode = []
    for a in range(q):
        for b in range(q):
            c = (a + b) % q
            d = (a + 2 * b) % q
            tetracode.append((a, b, c, d))
            
    initial_code_indices = []
    for h in tetracode:
        for x in range(q):
            for y in range(q):
                full_word = h + (x, y)
                idx = word_to_idx[full_word]
                initial_code_indices.append(idx)
                
    C = list(set(initial_code_indices)) # Ensure distinct
    C_set = set(C)
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Calculate initial cover counts
    for c in C:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    def can_remove(c_idx):
        """Check if removing c_idx leaves all words covered."""
        for nb in neighbors[c_idx]:
            if cover_count[nb] == 1:
                return False
        return True

    def greedy_reduce():
        """Remove any codeword that is not essential."""
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of C to allow removal during iteration
            # Optimization: only check candidates that might be removable?
            # For now, simple scan.
            for c in list(C):
                if can_remove(c):
                    # Remove c
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    C.remove(c)
                    C_set.remove(c)
                    changed = True

    def is_valid():
        return np.min(cover_count) >= 1

    # --- Simulated Annealing ---
    start_time = time.time()
    deadline = 118.0 # 2s margin for 120s limit
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    best_C = C.copy()
    best_cover_count = cover_count.copy()
    best_size = len(C)
    
    Lambda = 5.0
    
    # Initial greedy reduction
    greedy_reduce()
    if len(C) < best_size:
        best_size = len(C)
        best_C = C.copy()
        best_cover_count = cover_count.copy()

    # Precompute neighbor sets for faster intersection/difference if needed?
    # Neighbors are small (1 + n*(q-1) = 1 + 6*2 = 13).
    
    while time.time() < deadline:
        # Pick a random move
        # 80% Remove, 20% Swap
        if random.random() < 0.8:
            if not C:
                break
            c_old = random.choice(C)
            
            # Calculate newly uncovered
            newly_uncovered = 0
            for nb in neighbors[c_old]:
                if cover_count[nb] == 1:
                    newly_uncovered += 1
            
            delta_cost = -1.0 + Lambda * newly_uncovered
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                for nb in neighbors[c_old]:
                    cover_count[nb] -= 1
                C.remove(c_old)
                C_set.remove(c_old)
                
                # Greedy reduce to clean up
                greedy_reduce()
                
                # Update best if valid
                if is_valid():
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_cover_count = cover_count.copy()
        else:
            if not C:
                break
            c_old = random.choice(C)
            # Pick a random c_new not in C
            # To avoid infinite loop if C is full, but C is small.
            while True:
                c_new = random.randint(0, total_words - 1)
                if c_new not in C_set:
                    break
            
            # Perform swap on cover_count
            for nb in neighbors[c_old]:
                cover_count[nb] -= 1
            for nb in neighbors[c_new]:
                cover_count[nb] += 1
                
            # Check validity
            valid = np.min(cover_count) >= 1
            
            if valid:
                delta_cost = 0.0 
            else:
                # Count uncovered
                uncovered = np.sum(cover_count == 0)
                delta_cost = Lambda * uncovered
                
            if delta_cost == 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                C.remove(c_old)
                C_set.remove(c_old)
                C.append(c_new)
                C_set.add(c_new)
                
                # Greedy reduce
                greedy_reduce()
                
                if is_valid():
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_cover_count = cover_count.copy()
            else:
                # Reject: Revert cover_count
                for nb in neighbors[c_old]:
                    cover_count[nb] += 1
                for nb in neighbors[c_new]:
                    cover_count[nb] -= 1
                # C and C_set unchanged
                
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return best valid code found
    result = []
    for idx in best_C:
        result.append(list(words[idx]))
        
    return result
