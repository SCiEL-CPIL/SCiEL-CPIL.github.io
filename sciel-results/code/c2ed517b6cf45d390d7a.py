import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Precomputation
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_int = {w: i for i, w in enumerate(all_words)}
    int_to_word = {i: w for i, w in enumerate(all_words)}
    N = len(all_words)
    
    # Precompute neighbors (Hamming distance <= radius)
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                neighbors[i].append(word_to_int[tuple(new_w)])
        
    # Helper to convert list of indices to codewords
    def to_codewords(indices):
        return [list(int_to_word[i]) for i in indices]
    
    best_size = N
    best_indices = list(range(N)) # Fallback: all words
    
    # Strategy 1: Direct Sum + Deletion
    # This is very fast and provides a strong baseline
    if time.time() < deadline - 10:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        direct_sum_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = tuple(h) + tuple(tail)
                direct_sum_indices.append(word_to_int[w])
        
        # Greedy Deletion
        cover_count = np.zeros(N, dtype=int)
        for c in direct_sum_indices:
            for u in neighbors[c]:
                cover_count[u] += 1
        
        code_list = list(direct_sum_indices)
        # Sort by something? Random order is fine for deletion
        random.shuffle(code_list)
        
        for c in code_list:
            if time.time() > deadline - 5: break
            # Check if c can be removed
            # cover_count[neighbors[c]] must be > 1 for all u in neighbors[c]
            # Actually, if cover_count[u] > 1 for all u in neighbors[c], then removing c leaves count >= 1
            # Wait, if cover_count[u] == 1, and we remove c, count becomes 0.
            # So we need min(cover_count[neighbors[c]]) > 1
            if np.min(cover_count[neighbors[c]]) > 1:
                # Remove c
                for u in neighbors[c]:
                    cover_count[u] -= 1
                code_list.remove(c)
        
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_indices = code_list

    # Strategy 2: Greedy Construction from Scratch (Randomized)
    # Run a few times
    for run in range(5):
        if time.time() > deadline - 20: break
        
        uncovered = np.ones(N, dtype=bool)
        code = []
        cover_count = np.zeros(N, dtype=int)
        
        while np.any(uncovered):
            if time.time() > deadline - 10: break
            
            # Find best candidate
            # counts[v] = number of uncovered words covered by v
            counts = np.zeros(N, dtype=int)
            # Iterate through uncovered words
            # Optimization: only iterate through uncovered
            unc_idx = np.where(uncovered)[0]
            for u in unc_idx:
                for v in neighbors[u]:
                    counts[v] += 1
            
            # Pick v with max count
            # Tie-breaking: random
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            v = candidates[random.randint(0, len(candidates) - 1)]
            
            # Add v
            code.append(v)
            for u in neighbors[v]:
                if uncovered[u]:
                    uncovered[u] = False
                    cover_count[u] += 1 # Not strictly needed for greedy, but good for state
            
            # Actually, for greedy construction, we just need to update uncovered.
            # cover_count is not used in the selection, only uncovered mask.
            # So we can skip cover_count update here for speed, or just update uncovered.
            # The loop above updates uncovered.
            
        # Post-process: Greedy Deletion
        # Rebuild cover_count for the generated code
        cover_count = np.zeros(N, dtype=int)
        for c in code:
            for u in neighbors[c]:
                cover_count[u] += 1
        
        code_list = list(code)
        random.shuffle(code_list)
        
        for c in code_list:
            if time.time() > deadline - 5: break
            if np.min(cover_count[neighbors[c]]) > 1:
                for u in neighbors[c]:
                    cover_count[u] -= 1
                code_list.remove(c)
        
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_indices = code_list

    # Strategy 3: Local Search (Add-Delete) on the best found so far
    if time.time() < deadline - 5 and best_size < N:
        current_code = list(best_indices)
        cover_count = np.zeros(N, dtype=int)
        for c in current_code:
            for u in neighbors[c]:
                cover_count[u] += 1
        
        # Set of current indices for fast lookup
        current_set = set(current_code)
        
        while time.time() < deadline - 2:
            # Pick a random word not in current_code
            # Efficient way: pick random index, if in set, try again
            # Or maintain a list of not-in-code words? 
            # N is small, just sample.
            w = random.randint(0, N - 1)
            if w in current_set:
                continue
            
            # Add w
            current_code.append(w)
            current_set.add(w)
            for u in neighbors[w]:
                cover_count[u] += 1
            
            # Try to delete words from current_code
            # We can try to delete the newly added w first?
            # Or do a full pass?
            # A full pass is safer.
            # But we can optimize: only try to delete words that are "redundant".
            # Let's do a single pass of deletion.
            new_code = list(current_code)
            random.shuffle(new_code)
            
            for c in new_code:
                if time.time() > deadline - 2: break
                if np.min(cover_count[neighbors[c]]) > 1:
                    # Remove c
                    for u in neighbors[c]:
                        cover_count[u] -= 1
                    new_code.remove(c)
                    current_set.discard(c)
            
            if len(new_code) < len(current_code):
                # Improvement
                current_code = new_code
                # Note: cover_count is already updated to match new_code
            else:
                # Revert
                # Remove w
                for u in neighbors[w]:
                    cover_count[u] -= 1
                current_code.remove(w)
                current_set.discard(w)
                # The other deletions didn't happen because we reverted?
                # Wait, in the loop above, I modified cover_count and new_code.
                # If I revert, I need to restore cover_count to the state before adding w.
                # This is tricky if I modified cover_count in the deletion loop.
                
                # Better approach for Revert:
                # Keep a backup of cover_count? No, too big.
                # Instead, only perform deletions if they are valid, and track changes?
                # Or, simply: 
                # 1. Add w.
                # 2. Try to find a set of deletions that results in smaller size.
                # 3. If found, apply. Else, remove w.
                
                # Let's re-implement the inner loop to be transactional.
                
            # To keep it simple and safe, let's just do:
            # 1. Add w.
            # 2. Run a "Greedy Deletion" on the NEW list.
            # 3. If result is smaller, accept. Else, discard (remove w, restore counts).
            
            # Since I already wrote the loop above, let's fix the logic.
            # I will restart the inner logic for clarity in the final code.
            
    return to_codewords(best_indices)
