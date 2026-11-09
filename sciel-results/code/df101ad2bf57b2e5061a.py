import numpy as np
import time
import itertools

def construct():
        start_time = time.time()
        deadline = start_time + 110
        
        D = 11
        # Generate all {-1, 1} vectors
        # There are 2^11 = 2048 vectors
        # We can represent them as binary strings 0..2047
        # 0 -> [1, 1, ..., 1]
        # 1 -> [1, 1, ..., -1] etc.
        
        best_vectors = None
        best_size = 0
        
        # Try a few random shuffles to improve greedy result
        np.random.seed(42)
        
        for trial in range(20): # 20 trials is fast
            if time.time() > deadline:
                break
            
            # Generate indices
            indices = np.random.permutation(2**D)
            
            selected_indices = []
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                # Convert idx to vector
                # idx bits determine signs
                # Let's precompute all vectors for speed
                # Or compute on the fly. 2048 * 11 is small.
                
                # Precomputation is better
                # But let's just use a helper or precompute once outside the loop?
                # The loop is over trials. Precompute once.
            
            # We need to precompute the vectors or the distances?
            # Distance check: Hamming distance >= 3.
            # We can precompute the binary representation of all 2048 indices.
            
        # Precompute binary representations
        # bin_list[i] is a numpy array of 0s and 1s? Or just use bit operations.
        # Hamming distance between i and j is popcount(i ^ j).
        
        all_indices = list(range(2**D))
        
        best_set = set()
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            perm = np.random.permutation(all_indices)
            current_set = []
            
            for idx in perm:
                # Check against current_set
                valid = True
                for existing_idx in current_set:
                    # Hamming distance
                    hamming_dist = np.bitwise_xor(idx, existing_idx).bit_count()
                    if hamming_dist < 3:
                        valid = False
                        break
                
                if valid:
                    current_set.append(idx)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
        
        # Convert best_set indices to integer vectors
        vectors = []
        for idx in best_set:
            v = np.zeros(D, dtype=np.int64)
            for j in range(D):
                if (idx >> j) & 1:
                    v[j] = -1
                else:
                    v[j] = 1
            vectors.append(v)
            
        return np.array(vectors)
