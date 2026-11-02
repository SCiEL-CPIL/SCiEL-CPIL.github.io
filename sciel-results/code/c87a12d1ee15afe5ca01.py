    h_work = best_h.copy()
    current_score = best_score
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9999 # slower cooling to allow more exploration
    
    block_sizes = [2, 4, 8, 16, 32]
    
    while elapsed_ok():
        bsize = np.random.choice(block_sizes)
        if bsize > N:
            bsize = N
        start_idx = np.random.randint(0, N - bsize + 1)
        
        # Flip in place
        h_work[start_idx:start_idx+bsize] *= -1
        s_try = _score(h_work)
        
        if s_try < current_score:
            current_score = s_try
            if current_score < best_score:
                best_score = current_score
                best_h = h_work.copy()
        else:
            delta = s_try - current_score
            if delta < 0: delta = 0
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                current_score = s_try
            else:
                # Revert
                h_work[start_idx:start_idx+bsize] *= -1
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
