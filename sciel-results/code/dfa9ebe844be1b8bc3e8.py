import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate all Type B vectors: 4 non-zero entries, each +/- 1
    # Shape: (5280, 11)
    # We can generate them by iterating over combinations of 4 indices
    vectors = []
    indices = range(D)
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.int64)
    N = len(V)
    
    best_set = []
    best_size = 0
    
    # Greedy selection with random restarts
    # We try to find a large subset where max dot product <= 2
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        np.random.shuffle(V)
        
        selected = []
        selected_arr = []
        
        for i in range(N):
            v = V[i]
            if not selected_arr:
                selected.append(i)
                selected_arr.append(v)
                continue
                
            # Check against all selected vectors
            # Dot products
            # If any dot product > 2, reject
            # Using numpy for speed if selected_arr is large
            if len(selected_arr) > 50:
                sel_mat = np.array(selected_arr)
                dots = sel_mat.dot(v)
                if np.max(dots) > 2:
                    continue
            else:
                conflict = False
                for u in selected_arr:
                    if np.dot(u, v) > 2:
                        conflict = True
                        break
                if conflict:
                    continue
            
            selected.append(i)
            selected_arr.append(v)
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            # Early stop if we find a very large set?
            # Theoretical max?
            # If best_size > 1000, we can probably stop or continue carefully.
            # Let's just keep going until deadline.
            
    if not best_set:
        # Fallback to seed (Type A) if something went wrong, though Type A is valid
        # But Type A score is 220. Type B should be better.
        # If Type B fails, return Type A.
        vectors_a = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_a.append(v)
        return np.array(vectors_a, dtype=np.int64)

    result = V[best_set]
    return result
