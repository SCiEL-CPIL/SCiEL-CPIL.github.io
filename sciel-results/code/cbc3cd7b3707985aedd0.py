import numpy as np
import time
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit
    
    # Initialize with Reed-Muller code (32 words)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
            
    # Convert to numpy array for vectorized operations
    # Use uint16 to store 17-bit words
    code = np.array(words, dtype=np.uint16)
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    best_size = len(code)
    best_code = code.copy()
    
    def get_distances(candidates, code):
        """
        candidates: np.array of shape (M, 1) or (M,)
        code: np.array of shape (K,)
        Returns: np.array of shape (M, K) with Hamming distances
        """
        # XOR: (M, 1) ^ (1, K) -> (M, K)
        # We need to handle bit counts.
        # numpy doesn't have a direct bit_count for integers.
        # We can use a lookup table or unpack bits.
        # Unpacking bits:
        # (XOR >> i) & 1
        # Sum over i.
        
        # This is slow in pure numpy if we loop over 17 bits.
        # Alternative: Use a precomputed lookup table for 16-bit chunks?
        # Or just use Python loops if M*K is small.
        # With M=1000, K=256, M*K = 256,000.
        # Python loop over 256,000 items takes ~0.05s.
        # Numpy vectorization of bit count is tricky.
        
        # Let's stick to Python loops for the distance check but optimize the inner loop.
        pass

    # Given the complexity of vectorized bit count in numpy without C extensions,
    # and the small N, a well-optimized Python loop might be competitive.
    # However, the current program is slow because it checks one by one.
    
    # Let's try a different strategy: 
    # Use a set for O(1) lookups? No, distance is not O(1).
    
    # I will implement a fast greedy add using Python but with a larger batch and early exit.
    
    def can_add_fast(w, code_list):
        for c in code_list:
            if (w ^ c).bit_count() < MIN_DIST:
                return False
        return True

    # Phase 1: Greedy Add
    # We will try to add words in batches.
    code_list = code.tolist()
    
    # Pre-compute neighbors? No.
    
    # Use a priority queue? No.
    
    # Let's use the numpy approach for generating candidates, but check in Python.
    # Generate 5000 random candidates at once.
    
    while time.time() - start_time < time_limit * 0.7:
        batch_size = 5000
        candidates = np_rng.randint(0, 2**N, size=batch_size, dtype=np.uint16)
        
        added_any = False
        for w in candidates:
            w_int = int(w)
            if can_add_fast(w_int, code_list):
                code_list.append(w_int)
                code = np.array(code_list, dtype=np.uint16)
                added_any = True
                if len(code_list) > best_size:
                    best_size = len(code_list)
                    best_code = code.copy()
                    
        if not added_any:
            # If no words added, try local search or break
            # To avoid infinite loop, we can break or try flips
            # Let's try a few flips
            improved = False
            for _ in range(100):
                if time.time() - start_time > time_limit * 0.7:
                    break
                idx = rng.randrange(len(code_list))
                bit = rng.randrange(N)
                new_w = code_list[idx] ^ (1 << bit)
                # Check conflicts
                valid = True
                for j, c in enumerate(code_list):
                    if j != idx and (new_w ^ c).bit_count() < MIN_DIST:
                        valid = False
                        break
                if valid:
                    code_list[idx] = new_w
                    improved = True
                    if len(code_list) > best_size:
                        best_size = len(code_list)
                        best_code = np.array(code_list, dtype=np.uint16)
            if not improved:
                break

    # Phase 2: Local Search to optimize
    # ... (similar to before but faster)
    
    if len(best_code) >= len(code_list):
        return best_code.tolist()
    return code_list
