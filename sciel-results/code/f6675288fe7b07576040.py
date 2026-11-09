def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    N = 17
    MIN_D = 6
    SIZE = 1 << N
    
    # Precompute weights
    pop_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
    all_words = np.arange(SIZE, dtype=np.int64)
    weights = (pop_table[all_words & 0xFF] + 
               pop_table[(all_words >> 8) & 0xFF] + 
               pop_table[(all_words >> 16) & 0xFF])
    
    # Precompute masks with weight < MIN_D
    low_weight_masks = all_words[weights < MIN_D].astype(np.int64)
    
    def try_linear_code(order):
        """Try to build a linear code using candidates in the given order."""
        current_code = np.array([0], dtype=np.int64)
        basis = []
        
        for g in order:
            if time.time() > deadline:
                break
            if g == 0:
                continue
            
            xor_vals = g ^ current_code
            min_dist = np.min(weights[xor_vals])
            
            if min_dist >= MIN_D:
                basis.append(g)
                new_code = np.concatenate([current_code, current_code ^ g])
                current_code = new_code
                
                if len(basis) >= 8:
                    break
        
        return current_code, len(basis)
    
    # Try multiple orderings to find a good linear code
    best_code = None
    best_size = 0
    
    # Order 1: Natural order
    order1 = np.arange(1, SIZE)
    code1, size1 = try_linear_code(order1)
    if size1 > best_size:
        best_code = code1
        best_size = size1
    
    # Order 2: By decreasing weight
    order2 = np.argsort(-weights[1:]) + 1
    if time.time() < deadline - 5:
        code2, size2 = try_linear_code(order2)
        if size2 > best_size:
            best_code = code2
            best_size = size2
    
    # Order 3: Random shuffle
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        order3 = rng.permutation(np.arange(1, SIZE))
        code3, size3 = try_linear_code(order3)
        if size3 > best_size:
            best_code = code3
            best_size = size3
    
    if best_code is None:
        best_code = np.array([0], dtype=np.int64)
        best_size = 0
    
    # Now extend nonlinearly
    forbidden = np.zeros(SIZE, dtype=bool)
    
    # Mark balls around existing codewords
    for c in best_code:
        forbidden[c ^ low_weight_masks] = True
    
    added_set = set(best_code.tolist())
    
    # Greedy nonlinear extension: try to add words not in forbidden set
    # Process in order of decreasing weight (heuristic: high-weight words are "far" from low-weight ones)
    order = np.argsort(-weights)
    
    for w in order:
        if time.time() > deadline:
            break
        
        if not forbidden[w]:
            added_set.add(int(w))
            forbidden[w ^ low_weight_masks] = True
    
    return list(added_set)
