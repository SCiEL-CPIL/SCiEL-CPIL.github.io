import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    total_words = q ** n
    
    def index_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w

    # Precompute neighbor indices for all words
    # This is O(3^n * n * q) which is small for n=8, q=3
    neighbor_indices = []
    for idx in range(total_words):
        word = index_to_word(idx)
        neigh = [idx]
        for pos in range(n):
            original = word[pos]
            for val in range(q):
                if val == original:
                    continue
                word[pos] = val
                new_idx = 0
                for i in range(n):
                    new_idx += word[i] * (q ** i)
                neigh.append(new_idx)
            word[pos] = original
        neighbor_indices.append(neigh)
        
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * total_words
        code = []
        code_set = set()
        
        while True:
            if time.time() > deadline:
                break
                
            # Check if done
            # Using sum is faster than any for list of bools? No, any stops early.
            # But we need to know if ANY is true.
            # Actually, if we maintain a count of uncovered, it's O(1).
            # Let's maintain uncovered_count.
            
            if not any(uncovered):
                break
            
            best_candidate = -1
            best_count = -1
            
            # Iterate all possible candidates
            for c_idx in range(total_words):
                if c_idx in code_set:
                    continue
                # Count how many uncovered words this candidate covers
                count = 0
                for n_idx in neighbor_indices[c_idx]:
                    if uncovered[n_idx]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_candidate = c_idx
            
            if best_count == 0:
                break
                
            # Add best candidate
            code.append(best_candidate)
            code_set.add(best_candidate)
            
            # Mark covered
            for n_idx in neighbor_indices[best_candidate]:
                if uncovered[n_idx]:
                    uncovered[n_idx] = False
            
        return code

    def polish(code, seed):
        if not code:
            return code
        rng = random.Random(seed)
        code_set = set(code)
        
        # Initialize cover counts
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c_idx in code:
            for n_idx in neighbor_indices[c_idx]:
                cover_count[n_idx] += 1
        
        # Try to remove words
        code_list = list(code)
        rng.shuffle(code_list)
        
        for c_idx in code_list:
            if time.time() > deadline:
                break
            # Tentatively remove
            for n_idx in neighbor_indices[c_idx]:
                cover_count[n_idx] -= 1
            
            # Check if any word has count 0
            if np.min(cover_count) >= 1:
                # Safe to remove
                code_set.remove(c_idx)
            else:
                # Restore
                for n_idx in neighbor_indices[c_idx]:
                    cover_count[n_idx] += 1
        
        return list(code_set)

    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy restarts
    seed = 0
    while time.time() < deadline:
        code = greedy_cover(seed)
        code = polish(code, seed)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # If we reached 486, we can stop early? 
            # 486 is the known optimum.
            if best_size == 486:
                break
        
        seed += 1

    if best_code is None:
        # Fallback to all words (valid but large)
        return [index_to_word(i) for i in range(total_words)]
        
    return [index_to_word(c) for c in best_code]
