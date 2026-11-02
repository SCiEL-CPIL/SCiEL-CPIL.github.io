import numpy as np
import time

N = 400

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _sign_flip_search(x, time_limit):
    """
    Local search by flipping signs of elements to minimize the objective.
    """
    x = x.copy()
    s = np.sum(x)
    if abs(s) < 1e-10:
        # Perturb if sum is zero
        x[0] += 1.0
        s = np.sum(x)
        
    ac = _autoconv_fft(x)
    best_val = np.max(np.abs(ac)) / (s * s)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        best_flip_idx = -1
        best_new_val = best_val
        best_new_x = None
        
        # Try flipping each sign
        for i in range(N):
            x_trial = x.copy()
            x_trial[i] *= -1.0
            s_trial = s - 2.0 * x[i]
            
            if abs(s_trial) < 1e-10:
                continue
            
            ac_trial = _autoconv_fft(x_trial)
            val_trial = np.max(np.abs(ac_trial)) / (s_trial * s_trial)
            
            if val_trial < best_new_val:
                best_new_val = val_trial
                best_flip_idx = i
                best_new_x = x_trial
                improved = True
        
        if improved:
            x = best_new_x
            s = np.sum(x)
            ac = _autoconv_fft(x)
            best_val = best_new_val
        else:
            break
            
    return x, best_val

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 145  # safety margin
    
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    # Phase 1: Initial structured candidates
    initializations = [
        np.ones(N),
        np.ones(N) * (-1) ** np.arange(N), # Alternating signs
        np.concatenate([np.ones(N//2), -np.ones(N - N//2)]), # Block signs
    ]
    
    # Random sign initializations
    for _ in range(5):
        initializations.append(np.sign(rng.normal(0, 1, N)))
        
    # Add some structured magnitudes with random signs
    for _ in range(5):
        mags = rng.uniform(0.5, 1.5, N)
        signs = np.sign(rng.normal(0, 1, N))
        initializations.append(mags * signs)
        
    for x0 in initializations:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        x_opt, val_opt = _sign_flip_search(x0, time_limit=1.0) # 1s per init
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Phase 2: Continue optimizing the best found so far with more time
    remaining_time = TIME_LIMIT - (time.time() - START_TIME)
    if remaining_time > 2.0:
        x_opt, val_opt = _sign_flip_search(best_x, time_limit=remaining_time - 1.0)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt

    # Phase 3: Random restarts with perturbations if time remains
    remaining_time = TIME_LIMIT - (time.time() - START_TIME)
    while remaining_time > 5.0:
        # Perturb best solution: flip a random subset of signs
        x_pert = best_x.copy()
        num_flips = rng.integers(1, N // 4)
        indices = rng.choice(N, num_flips, replace=False)
        x_pert[indices] *= -1.0
        
        x_opt, val_opt = _sign_flip_search(x_pert, time_limit=remaining_time / 4.0)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
            
        remaining_time = TIME_LIMIT - (time.time() - START_TIME)

    return best_x
