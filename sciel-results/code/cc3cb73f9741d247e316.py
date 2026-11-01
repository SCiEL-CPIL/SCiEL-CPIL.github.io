import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Generate all integer vectors with exactly 4 non-zero entries, each +/-1
    # Shape: (5280, 11)
    vecs = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for i, s in zip(combo, signs):
                v[i] = s
            vecs.append(v)
    
    vecs = np.array(vecs, dtype=np.int64)
    n_vecs = len(vecs)
    
    best_k = 0
    best_vectors = None
    
    # Fallback to seed logic if no weight-4 found quickly or if it's bad
    # Seed logic: weight 2 vectors
    seed_vecs = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vecs.append(v)
    seed_array = np.array(seed_vecs, dtype=np.int64)
    best_k = len(seed_array)
    best_vectors = seed_array
    
    # Greedy selection for weight 4
    # We will try a few random shuffles
    rng = np.random.default_rng(42)
    
    max_trials = 100 # Adjust based on speed
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
            
        # Shuffle indices
        idx = rng.permutation(n_vecs)
        shuffled_vecs = vecs[idx]
        
        accepted = []
        # For efficiency, keep accepted in a list, and maybe a matrix for vectorized dot
        # But vstack is slow.
        # Alternative: Keep a list of numpy arrays.
        # Check: max dot <= 2
        
        # To speed up, we can batch check? No, greedy is sequential.
        # However, we can pre-filter?
        
        # Simple loop
        # Optimization: If accepted is empty, add first.
        # Use a list of vectors.
        acc_list = []
        
        for v in shuffled_vecs:
            if not acc_list:
                acc_list.append(v)
                continue
                
            # Check against all accepted
            # Convert v to array if not already (it is a row from numpy array, so it is)
            # acc_list contains numpy arrays of shape (D,)
            # We can stack them for dot product? 
            # Creating a stack every time is expensive.
            # Better: Keep a matrix `A` and update it.
            # But vstack creates a new array.
            # Alternative: Keep list, and only vectorize if list is small?
            # Or just use a loop? 11D dot product is very fast in Python?
            # No, Python loop over 1000 items is ~0.1ms. 5000 items ~0.5ms.
            # 5000 candidates * 0.5ms = 2.5s per trial.
            # 100 trials = 250s. Too slow.
            
            # We need vectorized dot product.
            # Maintain `A` as a numpy array.
            # `A` shape (k, D).
            # `dots = A @ v`
            # `if dots.max() <= 2: A = np.vstack((A, v))`
            
            # vstack is O(k). Total O(k^2).
            # If k=1000, 1000*1000*11 = 11M ops.
            # 5000 candidates.
            # This is heavy.
            
            # Alternative:
            # Use a list of vectors.
            # Periodically convert to matrix?
            # Or just accept the cost if we reduce trials.
            
            pass # Placeholder for logic below

    # Re-implementing the loop with proper efficiency
    best_k = 0
    best_vectors = seed_array
    
    for trial in range(50): # Reduced trials
        if time.time() > deadline:
            break
            
        idx = rng.permutation(n_vecs)
        shuffled_vecs = vecs[idx]
        
        A = None
        k = 0
        
        for v in shuffled_vecs:
            if A is None:
                A = v.reshape(1, -1)
                k = 1
                continue
            
            # Check compatibility
            # A is (k, D), v is (D,)
            dots = A @ v
            if dots.max() <= 2:
                A = np.vstack((A, v))
                k += 1
        
        if k > best_k:
            best_k = k
            best_vectors = A

    return best_vectors
