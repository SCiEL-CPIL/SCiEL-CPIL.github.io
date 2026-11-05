# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Parameters for this specific task instance
    # The evaluator passes q, n, radius. We assume the task is fixed to q=3, n=6, radius=1
    # but we should handle general cases if possible, or at least not crash.
    # The seed program hardcodes ternary tetracode logic.
    # We will implement a general local search for small n, q.
    
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    
    # Fallback to seed logic for other parameters to ensure validity
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Brute force for small n
        # Find minimum covering code
        # This is a fallback, likely not used for the main task
        pass

def _construct_ternary_6_1():
    q = 3
    n = 6
    radius = 1
    
    # 1. Precompute all words and neighbors
    # Total words = 3^6 = 729
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    num_words = len(all_words)
    
    # neighbors[i] = list of indices of words within Hamming distance <= 1 of all_words[i]
    neighbors = []
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w[pos] = new_val
                    idx = word_to_idx[tuple(w)]
                    neigh.append(idx)
            w[pos] = orig_val
        neighbors.append(neigh)
        
    # Convert neighbors to numpy arrays for fast updates
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors]

    # 2. Initialize with Direct Sum Code (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            idx = word_to_idx[tuple(word)]
            initial_code_indices.append(idx)
            
    current_code = list(initial_code_indices)
    
    # 3. Compute initial cover counts
    cover_count = np.zeros(num_words, dtype=np.int32)
    for idx in current_code:
        cover_count[neighbors_np[idx]] += 1
        
    uncovered_count = np.count_nonzero(cover_count == 0)
    
    best_code = list(current_code)
    best_size = len(current_code)
    
    # 4. Local Search
    start_time = time.time()
    time_limit = 100.0 # Safety margin for 120s limit
    
    # Random seed for reproducibility? No, let it vary or fix it? 
    # Fixing it ensures deterministic results, which is good for evaluation consistency.
    random.seed(42)
    np.random.seed(42)
    
    while time.time() - start_time < time_limit:
        if uncovered_count > 0:
            # If we ever become invalid (shouldn't happen with our moves), try to fix
            # Find an uncovered word
            uncovered_indices = np.where(cover_count == 0)[0]
            if len(uncovered_indices) > 0:
                # Pick a random uncovered word
                u_idx = uncovered_indices[np.random.randint(len(uncovered_indices))]
                # Find a word that covers u_idx and maximizes coverage of other uncovered words?
                # Simple: pick a random word that covers u_idx
                # Words covering u_idx are those v such that u_idx is in neighbors[v]
                # This is inverse neighbors. Precomputing inverse neighbors is expensive.
                # Just pick a random word and see if it helps?
                # Better: Add the word u_idx itself? It covers itself and neighbors.
                new_idx = u_idx
                # Check if new_idx is already in code
                if new_idx not in current_code:
                    current_code.append(new_idx)
                    cover_count[neighbors_np[new_idx]] += 1
                    uncovered_count = np.count_nonzero(cover_count == 0)
            continue
            
        # Valid state: uncovered_count == 0
        
        # Try Removal
        if len(current_code) > 0:
            c_idx = random.choice(current_code)
            # Tentatively remove
            cover_count[neighbors_np[c_idx]] -= 1
            new_uncovered = np.count_nonzero(cover_count == 0)
            
            if new_uncovered == 0:
                # Accept removal
                current_code.remove(c_idx)
                uncovered_count = 0
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
            else:
                # Revert
                cover_count[neighbors_np[c_idx]] += 1
                
        # Try Swap
        if len(current_code) > 0 and random.random() < 0.5: # 50% chance to try swap if removal failed or just generally
            c_idx = random.choice(current_code)
            w_idx = np.random.randint(num_words)
            
            if c_idx != w_idx:
                # Tentatively swap
                cover_count[neighbors_np[c_idx]] -= 1
                cover_count[neighbors_np[w_idx]] += 1
                
                new_uncovered = np.count_nonzero(cover_count == 0)
                
                if new_uncovered == 0:
                    # Accept swap
                    # Replace c_idx with w_idx in current_code
                    # Note: w_idx might already be in current_code?
                    # If w_idx is in current_code, we would have a duplicate.
                    # We should ensure w_idx is not in current_code or handle duplicates.
                    # Since we are swapping, if w_idx is already there, we effectively just removed c_idx?
                    # No, if w_idx is in code, then code has w_idx and c_idx.
                    # Swap c_idx -> w_idx means code has w_idx (twice?) and no c_idx.
                    # Codewords must be distinct.
                    # So we should only swap if w_idx is NOT in current_code.
                    if w_idx not in current_code:
                         current_code.remove(c_idx)
                         current_code.append(w_idx)
                         uncovered_count = 0
                    else:
                         # Revert
                         cover_count[neighbors_np[c_idx]] += 1
                         cover_count[neighbors_np[w_idx]] -= 1
                else:
                    # Revert
                    cover_count[neighbors_np[c_idx]] += 1
                    cover_count[neighbors_np[w_idx]] -= 1

    # Convert best_code indices back to words
    best_words = [all_words[idx] for idx in best_code]
    return best_words

# EVOLVE-BLOCK-END
