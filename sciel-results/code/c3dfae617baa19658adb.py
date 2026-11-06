import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Evaluator kills at 90s. Use 85s deadline for safety.
    deadline = start_time + 85.0
    
    N = 17
    MIN_DIST = 6
    
    # Precompute bit masks for vectorized operations
    # Using uint8 for matrix storage to save memory and speed up XOR/SUM
    bit_masks = np.array([1 << i for i in range(N)], dtype=np.uint32)
    
    # Helper: Convert list of ints to bit matrix (M x N)
    def ints_to_mat(words):
        if len(words) == 0:
            return np.empty((0, N), dtype=np.uint8)
        arr = np.array(words, dtype=np.uint32)
        mat = ((arr[:, None] & bit_masks[None, :]) > 0).astype(np.uint8)
        return mat
    
    # Helper: Check if a single word (int) is valid against a matrix
    # Returns True if min distance >= MIN_DIST
    def is_valid(word_int, mat):
        if len(mat) == 0:
            return True
        w = ((word_int & bit_masks) > 0).astype(np.uint8)
        # Vectorized distance calculation
        dists = (mat ^ w).sum(axis=1)
        return bool(np.min(dists) >= MIN_DIST)
    
    # Helper: Add a word to the code, updating the matrix
    def add_word(words, mat, word_int):
        words.append(word_int)
        w = ((word_int & bit_masks) > 0).astype(np.uint8)
        if len(mat) == 0:
            return w
        return np.vstack([mat, w])
    
    # Strategy: Systematic Greedy in Gray Code Order
    # This explores the hypercube in a way that local changes are frequent,
    # which can help find valid words that are "near" existing ones but distinct enough.
    # Alternatively, we can just iterate 0 to 2^N-1. Gray code is better for locality.
    
    # Let's use a simple integer iteration first, but optimized.
    # Actually, the "inspiration" mentioned a fast greedy search that explores the entire space.
    # Iterating 0..131071 is fast if the check is vectorized.
    
    # We will perform multiple passes or a single exhaustive pass.
    # A single exhaustive pass in natural order (0, 1, 2, ...) is a valid greedy strategy.
    # To improve, we can randomize the order in subsequent runs if time permits,
    # but the first run should be deterministic and fast.
    
    best_words = []
    best_mat = np.empty((0, N), dtype=np.uint8)
    
    # Run 1: Exhaustive greedy in natural order
    # This is very fast because we can vectorize the check if we batch, 
    # but checking one by one is O(M*N) per word. M grows up to ~300. 
    # 131k * 300 * 17 ~ 670M ops. NumPy can handle this in < 1s per pass?
    # Let's try.
    
    current_words = []
    current_mat = np.empty((0, N), dtype=np.uint8)
    
    # We can speed this up by maintaining a "forbidden" region or just checking.
    # Let's just check.
    
    for i in range(2**N):
        if time.time() > deadline - 2.0:
            break
        
        # Optimization: If we have a lot of words, checking is expensive.
        # But 131k iterations is small.
        
        # Check validity
        if len(current_mat) == 0:
            valid = True
        else:
            w = ((i & bit_masks) > 0).astype(np.uint8)
            # Calculate distances to all existing words
            dists = (current_mat ^ w).sum(axis=1)
            valid = bool(np.min(dists) >= MIN_DIST)
            
        if valid:
            current_words.append(i)
            w = ((i & bit_masks) > 0).astype(np.uint8)
            if len(current_mat) == 0:
                current_mat = w[None, :]
            else:
                current_mat = np.vstack([current_mat, w])
                
    if len(current_words) > len(best_words):
        best_words = current_words
        best_mat = current_mat
    
    # If we have time, try randomized greedy restarts
    # Random order often finds larger codes than natural order because it avoids "traps" 
    # where natural order fills up a cluster too densely, preventing other clusters from forming.
    
    while time.time() < deadline - 5.0:
        # Shuffle the order of candidates
        candidates = list(range(2**N))
        random.shuffle(candidates)
        
        current_words = []
        current_mat = np.empty((0, N), dtype=np.uint8)
        
        # We don't need to check all 131k if we get stuck? 
        # No, greedy must check all to be "exhaustive greedy".
        # But we can break early if we are confident? No, just run it.
        # 131k checks is fast.
        
        for i in candidates:
            if time.time() > deadline - 2.0:
                break
                
            if len(current_mat) == 0:
                valid = True
            else:
                w = ((i & bit_masks) > 0).astype(np.uint8)
                dists = (current_mat ^ w).sum(axis=1)
                valid = bool(np.min(dists) >= MIN_DIST)
                
            if valid:
                current_words.append(i)
                w = ((i & bit_masks) > 0).astype(np.uint8)
                if len(current_mat) == 0:
                    current_mat = w[None, :]
                else:
                    current_mat = np.vstack([current_mat, w])
                    
        if len(current_words) > len(best_words):
            best_words = current_words
            best_mat = current_mat
            
        # If we found a very large code, we might stop early?
        # Theoretical max is ~340. If we hit >300, it's good.
        # But let's keep trying until deadline.
        
    return best_words

