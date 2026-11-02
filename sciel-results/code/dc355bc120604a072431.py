import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave margin for 150s limit
    
    best_vectors = []
    best_score = 0
    
    # We will try different weights W
    # Weight 2 is the seed (220). We expect better.
    for W in [3, 4, 5, 6]:
        if time.time() > deadline - 20: # Need 20s for generation/checking
            break
            
        # Generate candidates
        candidates = []
        # Indices for W non-zero elements
        for indices in itertools.combinations(range(D), W):
            # Signs
            for signs in itertools.product([1, -1], repeat=W):
                v = [0] * D
                for idx, s in zip(indices, signs):
                    v[idx] = s
                candidates.append(v)
        
        candidates = np.array(candidates, dtype=np.int64)
        
        # Try a few restarts for this W
        best_for_w = []
        num_restarts = 5
        # Reduce restarts if time is tight
        if time.time() > deadline - 10:
             num_restarts = 1

        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.shuffle(candidates)
            selected = []
            selected_arr = [] # List of numpy arrays or a growing array? 
            # Using a list of numpy arrays is fine, we convert to array for vectorized check
            
            for v in candidates:
                if time.time() > deadline - 2:
                    break
                
                if not selected:
                    selected.append(v)
                    selected_arr = [v]
                    continue
                
                # Check against all selected
                # Convert selected to array for vectorized op
                # This is inefficient if done every step.
                # Better: keep selected as list, and if len > threshold, convert to array.
                # Or just use list comprehension for small D.
                # D=11 is small.
                
                # Let's use a simple loop for clarity and speed for small D
                # dist_sq = sum((v[i]-s[i])**2 for i in range(D))
                
                valid = True
                for s in selected:
                    # Optimization: if we store as tuple, access is fast
                    # v and s are numpy arrays or lists.
                    # Let's use lists for v and s to avoid numpy overhead in inner loop?
                    # No, numpy is faster for vector ops.
                    # But creating array every time is slow.
                    
                    # Let's stick to lists for inner loop if K is small.
                    # Or use numpy array for selected block.
                    
                    pass # Placeholder for logic below

            # Re-implementing the inner loop properly
            # We'll use a list of tuples for selected vectors for fast access
            # and compute dist manually or via numpy on the fly?
            # Given D=11, manual loop in Python might be slower than numpy vectorization on a batch.
            # But we are checking one by one.
            
            # Let's use a numpy array for selected vectors, appending to it.
            # Appending to numpy array is slow.
            # Use list of numpy arrays.
            
            selected_list = []
            for v in candidates:
                if time.time() > deadline - 2:
                    break
                
                if not selected_list:
                    selected_list.append(v)
                    continue
                
                # Check against all in selected_list
                # We can vectorize this if we keep selected_list as a numpy array
                # But it grows.
                # Let's just loop. D=11.
                ok = True
                for s in selected_list:
                    # dist sq
                    # v and s are 1D numpy arrays
                    d = np.sum((v - s) ** 2)
                    if d < W: # Need >= W
                        ok = False
                        break
                if ok:
                    selected_list.append(v)
            
            if len(selected_list) > len(best_for_w):
                best_for_w = selected_list

        if len(best_for_w) > best_score:
            best_score = len(best_for_w)
            best_vectors = best_for_w

    # If we found nothing (should not happen), return seed
    if not best_vectors:
         # Fallback to seed logic
         vectors = []
         for i, j in itertools.combinations(range(D), 2):
             for si in (1, -1):
                 for sj in (1, -1):
                     v = [0] * D
                     v[i], v[j] = si, sj
                     vectors.append(v)
         return np.array(vectors, dtype=np.int64)
         
    return np.array(best_vectors, dtype=np.int64)
