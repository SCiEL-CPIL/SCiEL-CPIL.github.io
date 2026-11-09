def construct():
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 85
    
    N = 17
    MIN_D = 6
    SIZE = 1 << N
    
    # Precompute weights
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)
    
    def get_linear_code():
        # Numpy-accelerated greedy linear construction
        current_code = np.array([0], dtype=np.int32)
        basis = []
        
        # Process candidates in order, but use numpy for distance checks
        for w in range(1, SIZE):
            if len(basis) == 8:
                break
            if time.time() > deadline - 50:
                break
            
            # Check distance to all current codewords using numpy
            dists = weights[w ^ current_code]
            if dists.min() >= MIN_D:
                basis.append(w)
                new_words = current_code ^ w
                current_code = np.concatenate([current_code, new_words])
                
        return current_code.tolist()

    # Step 1: Get the 256-word linear code
    lin_code = get_linear_code()
    
    if len(lin_code) < 256:
        # Fallback: return what we have
        return lin_code
    
    # Step 2: Find candidates outside the linear code
    lin_code_np = np.array(lin_code, dtype=np.int32)
    
    # Vectorized: check all words at once in batches
    candidates = []
    batch_size = 4096
    for start in range(0, SIZE, batch_size):
        end = min(start + batch_size, SIZE)
        batch = np.arange(start, end, dtype=np.int32)
        # Compute distances from each word in batch to all lin_code words
        # This is (batch_size, 256) array of XORs, then weight lookup
        # Too large: 4096 * 256 = 1M elements. Do it in smaller sub-batches.
        sub_batch_size = 512
        for sb_start in range(0, batch_size, sub_batch_size):
            sb_end = min(sb_start + sub_batch_size, batch_size)
            sub = batch[sb_start:sb_end]
            # (sub_len, 256) XOR
            xors = sub[:, None] ^ lin_code_np[None, :]
            # Flatten for weight lookup
            wts = weights[xors.flatten()]
            wts = wts.reshape(-1, len(lin_code))
            min_dists = wts.min(axis=1)
            valid_idx = np.where(min_dists >= MIN_D)[0]
            for idx in valid_idx:
                word = int(sub[idx])
                if word not in lin_code_set:
                    candidates.append(word)
        
        if time.time() > deadline - 15:
            break
    
    if not candidates:
        return lin_code

    # Step 3: Find maximum subset of candidates with pairwise distance >= 6
    # Use numpy-accelerated greedy with multiple restarts
    
    max_found = 0
    best_set = []
    
    num_restarts = 200
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        cands = candidates[:]
        random.shuffle(cands)
        
        current_set = []
        for w in cands:
            if current_set:
                # Check distance to all in current_set using numpy
                cur_np = np.array(current_set, dtype=np.int32)
                dists = weights[w ^ cur_np]
                if dists.min() < MIN_D:
                    continue
            current_set.append(w)
                
        if len(current_set) > max_found:
            max_found = len(current_set)
            best_set = current_set[:]
            
    final_code = lin_code + best_set
    return final_code
