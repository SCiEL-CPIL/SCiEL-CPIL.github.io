def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580
    
    best_points = np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    best_k = 2
    
    norms_to_try = [8, 10, 12, 4]
    
    for R in norms_to_try:
        if time.time() > deadline:
            break
            
        if best_k >= 500: 
            break

        try:
            candidates = generate_vectors_norm(R, d)
        except Exception:
            continue
            
        n_cands = len(candidates)
        if n_cands < best_k:
            continue
            
        threshold = R // 2
        
        # Pre-allocate buffer
        sel_arr = np.empty((n_cands, d), dtype=np.int64)
        
        while time.time() < deadline - 5:
            idx = np.random.permutation(n_cands)
            shuffled_candidates = candidates[idx]
            
            k = 0
            sel_arr[k] = shuffled_candidates[0]
            k += 1
            
            for i in range(1, n_cands):
                v = shuffled_candidates[i]
                if k > 0:
                    dots = sel_arr[:k] @ v
                    if np.all(dots <= threshold):
                        sel_arr[k] = v
                        k += 1
                        
            if k > best_k:
                best_k = k
                best_points = sel_arr[:k].copy()
                
    return best_points
