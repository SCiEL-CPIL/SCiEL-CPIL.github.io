import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate all weight-2 vectors
    vecs_w2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vecs_w2.append(v)
    
    # 2. Generate all weight-8 vectors
    vecs_w8 = []
    for indices in itertools.combinations(range(D), 8):
        for signs in itertools.product([1, -1], repeat=8):
            v = [0] * D
            for k, idx in enumerate(indices):
                v[idx] = signs[k]
            vecs_w8.append(v)
    
    # Convert to numpy arrays for faster dot products if needed, 
    # but list of lists is fine for small counts.
    # We will maintain a list of selected weight-8 vectors.
    selected_w8 = []
    
    # Shuffle weight-8 vectors
    np.random.shuffle(vecs_w8)
    
    # Greedy selection
    for v in vecs_w8:
        if time.time() > deadline:
            break
            
        # Check compatibility with all selected_w8
        # Condition: dot product <= 4
        compatible = True
        for u in selected_w8:
            # Dot product
            dot = sum(a*b for a, b in zip(v, u))
            if dot > 4:
                compatible = False
                break
        
        if compatible:
            selected_w8.append(v)
            
    # Combine
    all_vecs = vecs_w2 + selected_w8
    return np.array(all_vecs, dtype=np.int64)
