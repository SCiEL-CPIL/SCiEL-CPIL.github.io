import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy or random
        pass 

    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Precompute all words and their neighbors
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words) # 729
    
    # Precompute neighbors for each word (Hamming distance <= 1)
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # self
        w_list = list(w)
        for pos in range(6):
            orig = w_list[pos]
            for val in range(3):
                if val != orig:
                    w_list[pos] = val
                    neighbors[i].append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
    
    # Initialize with the 81-word solution
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_codewords.append(tuple(h + list(tail)))
    
    C = [word_to_idx[w] for w in initial_codewords]
    
    # Initialize cover_count
    cover_count = np.zeros(N, dtype=np.int32)
    for c in C:
        for nb in neighbors[c]:
            cover_count[nb] += 1
    
    def is_valid():
        return np.min(cover_count) > 0
    
    def try_remove(idx_in_C):
        """Tries to remove C[idx_in_C]. Returns True if successful."""
        c = C[idx_in_C]
        # Tentatively remove
        for nb in neighbors[c]:
            cover_count[nb] -= 1
        
        if is_valid():
            # Accept removal
            C.pop(idx_in_C)
            return True
        else:
            # Revert
            for nb in neighbors[c]:
                cover_count[nb] += 1
            return False

    def greedy_remove():
        """Iteratively remove any removable codeword."""
        changed = True
        while changed and (time.time() - start_time) < deadline - 5.0:
            changed = False
            # Shuffle to avoid bias
            indices = list(range(len(C)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(C): continue
                if try_remove(i):
                    changed = True
                    # Restart loop if we want to be thorough, or continue
                    # Continuing is faster, but might miss cascading removals if order matters?
                    # Actually, if we remove one, counts change, so subsequent checks are valid.
                    # But the list `indices` is based on old length.
                    # We broke the index alignment.
                    # Better to restart the while loop or break the for loop.
                    break 

    # Phase 1: Greedy Removal
    greedy_remove()
    
    best_size = len(C)
    best_C = C[:]
    best_cover = cover_count.copy()
    
    # Phase 2: Local Search with Swaps
    # We will try to swap a codeword with another to enable further removals.
    
    while (time.time() - start_time) < deadline - 10.0:
        if len(C) < best_size:
            best_size = len(C)
            best_C = C[:]
            best_cover = cover_count.copy()
            
        # Pick a random codeword to swap
        if not C:
            break
            
        i = random.randint(0, len(C) - 1)
        c_old = C[i]
        
        # Generate a candidate for c_new
        # Strategy: Pick a random word, or a word that covers many "weak" words
        # Weak words are those with cover_count == 1
        # For now, just random sampling is often enough with many iterations
        c_new = random.randint(0, N - 1)
        
        if c_new == c_old:
            continue
            
        # Check if c_new is already in C
        # This is O(|C|). |C| is small (~70-80).
        if c_new in C:
            continue
            
        # Tentatively swap: Remove c_old, Add c_new
        for nb in neighbors[c_old]:
            cover_count[nb] -= 1
        for nb in neighbors[c_new]:
            cover_count[nb] += 1
            
        if is_valid():
            # State is valid. Try to reduce size.
            C[i] = c_new
            # Run greedy removal
            greedy_remove()
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C[:]
                best_cover = cover_count.copy()
            else:
                # Revert to state before swap if we didn't improve
                # We need to revert the swap AND the greedy removals
                # This is tricky because greedy_remove modifies C and cover_count.
                # Easier: Save state before swap, restore if no improvement.
                # Let's restructure:
                # Save C and cover_count before swap.
                pass
        else:
            # Revert
            for nb in neighbors[c_old]:
                cover_count[nb] += 1
            for nb in neighbors[c_new]:
                cover_count[nb] -= 1

    # The revert logic above is flawed because greedy_remove changes the state significantly.
    # We need a proper "save/restore" mechanism.
    
    # Let's restart the search loop with proper state management.
    
    C = best_C[:]
    cover_count = best_cover.copy()
    
    def save_state():
        return C[:], cover_count.copy()
        
    def restore_state(state):
        nonlocal C, cover_count
        C, cover_count = state[0], state[1]
        
    # Re-run greedy remove to ensure we are at a local minimum
    greedy_remove()
    
    best_size = len(C)
    best_C = C[:]
    best_cover = cover_count.copy()
    
    while (time.time() - start_time) < deadline - 5.0:
        if len(C) < best_size:
            best_size = len(C)
            best_C = C[:]
            best_cover = cover_count.copy()
            
        if not C:
            break
            
        i = random.randint(0, len(C) - 1)
        c_old = C[i]
        
        # Candidate generation:
        # 50% random, 50% biased towards covering weak spots
        if random.random() < 0.5:
            c_new = random.randint(0, N - 1)
        else:
            # Find words with cover_count == 1
            weak_indices = np.where(cover_count == 1)[0]
            if len(weak_indices) > 0:
                target_word = weak_indices[random.randint(0, len(weak_indices) - 1)]
                # Pick a random neighbor of the target word as candidate
                c_new = neighbors[target_word][random.randint(0, len(neighbors[target_word]) - 1)]
            else:
                c_new = random.randint(0, N - 1)
                
        if c_new == c_old or c_new in C:
            continue
            
        state_before = save_state()
        
        # Swap
        for nb in neighbors[c_old]:
            cover_count[nb] -= 1
        for nb in neighbors[c_new]:
            cover_count[nb] += 1
            
        if is_valid():
            C[i] = c_new
            greedy_remove()
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C[:]
                best_cover = cover_count.copy()
            else:
                restore_state(state_before)
        else:
            restore_state(state_before)
            
    # Return the best found
    return [list(all_words[i]) for i in best_C]
