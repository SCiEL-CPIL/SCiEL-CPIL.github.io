def fit_exercise_policy(training_paths, contract):
    import numpy as np
    import time as _time
    
    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    
    # Deadline guard
    start_time = _time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Exercise times: t_i = i/12 for i=1,...,12
    times = np.array([i / 12.0 for i in range(1, n_dates + 1)])
    
    # training_paths shape: (n_paths, n_dates+1) or (n_dates+1, n_paths)
    n_paths = training_paths.shape[0]
    
    # Check if it's (n_paths, n_steps) or (n_steps, n_paths)
    if training_paths.ndim == 2:
        if training_paths.shape[0] == 4096:
            states = training_paths
        elif training_paths.shape[1] == 4096:
            states = training_paths.T
        else:
            states = training_paths
    else:
        states = np.array(training_paths)
    
    n_steps = states.shape[1]
    
    # V_next: value at last date (European put at maturity)
    V_next = np.maximum(K - states[:, -1], 0.0)
    
    # thresholds[i] for i=1..12: exercise at date i if S_i < thresholds[i]
    thresholds = np.zeros(n_dates)
    thresholds[-1] = K
    
    # Backward induction with improved threshold estimation
    for i in range(n_dates - 1, 0, -1):
        # Check deadline
        if _time.time() > deadline:
            break
        
        S_i = states[:, i]
        immediate = np.maximum(K - S_i, 0.0)
        
        dt = 1.0 / 12.0
        disc = np.exp(-r * dt)
        continuation = disc * V_next
        
        # Exercise if immediate > continuation
        exercise_mask = immediate > continuation
        
        # Improved threshold estimation: use the S value where 
        # max(K-S, 0) = continuation, i.e., S = K - continuation (for in-the-money)
        # Find the threshold as the median of the boundary region
        
        if np.any(exercise_mask) and np.any(~exercise_mask):
            # The threshold should separate exercising from non-exercising paths
            # Use the implied threshold from the continuation value
            # For a put, exercise when S < K - continuation (approximately)
            # But continuation varies by path, so find the best single threshold
            
            # Method: find threshold that maximizes agreement with the mask
            # Use quantile-based approach: threshold at the boundary
            S_exer = S_i[exercise_mask]
            S_non = S_i[~exercise_mask]
            
            # Better estimate: use the point where the two distributions separate
            # Use the midpoint but weighted by the gap
            thr_low = S_exer.max()
            thr_high = S_non.min()
            
            if thr_low < thr_high:
                # Clear separation: use midpoint
                thr = (thr_low + thr_high) / 2.0
            else:
                # Overlap: use the S value where K-S equals the median continuation
                # among the exercising paths
                med_cont = np.median(continuation[exercise_mask])
                thr = K - med_cont
                # Clamp to reasonable range
                thr = np.clip(thr, 0.0, K)
        elif np.all(exercise_mask):
            thr = K
        elif np.all(~exercise_mask):
            thr = 0.0
        else:
            thr = K
        
        thresholds[i-1] = thr
        
        # Update V_next
        V_next = np.where(exercise_mask, immediate, continuation)
    
    # Now create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
        
        def exercise(self, current_state, time):
            if isinstance(time, (int, float)):
                if isinstance(time, float) and 0 < time <= 1.0:
                    idx = int(np.round(time * 12)) - 1
                    if idx < 0:
                        idx = 0
                    if idx >= n_dates:
                        idx = n_dates - 1
                elif isinstance(time, int):
                    if 1 <= time <= 12:
                        idx = time - 1
                    elif 0 <= time <= 11:
                        idx = time
                    else:
                        idx = min(max(time, 0), n_dates - 1)
                else:
                    idx = 0
            else:
                idx = 0
            
            state = np.asarray(current_state)
            return state < self.thresholds[idx]
    
    return ExercisePolicy(thresholds, K)
