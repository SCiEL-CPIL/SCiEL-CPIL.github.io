import numpy as np
import time

N = 400
DEADLINE = 118.0
M = 4 * N  # fine grid size
dx = 0.5 / M  # domain length is 0.5

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the step function.
    Uses FFT on a fine grid for efficiency.
    """
    f_fine = np.repeat(signed_step.astype(np.float64), M // N)
    fft_f = np.fft.rfft(f_fine, n=2 * M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
    autoconv *= dx
    return np.max(np.abs(autoconv))

def _run_sa(initial_step, start_time, deadline, seed):
    """Run simulated annealing from an initial configuration."""
    np.random.seed(seed)
    
    current_step = initial_step.copy()
    current_c3 = _compute_c3(current_step)
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # SA parameters
    T = 0.01  # initial temperature (small since we start near optimum)
    T_min = 1e-6
    alpha = 0.9999  # cooling rate
    
    no_improve = 0
    patience = 5000
    
    while time.time() < deadline - 3.0:
        # Choose move: single flip (70%) or multi-flip 2-3 (30%)
        r = np.random.random()
        if r < 0.7:
            idx = np.random.randint(N)
            old_val = current_step[idx]
            current_step[idx] = -old_val
            new_c3 = _compute_c3(current_step)
            delta = new_c3 - current_c3
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_c3 = new_c3
                if current_c3 < best_c3:
                    best_c3 = current_c3
                    best_step = current_step.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_step[idx] = old_val
                no_improve += 1
        else:
            n_flips = np.random.randint(2, 4)
            idxs = np.random.choice(N, size=n_flips, replace=False)
            old_vals = current_step[idxs].copy()
            current_step[idxs] = -old_vals
            new_c3 = _compute_c3(current_step)
            delta = new_c3 - current_c3
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_c3 = new_c3
                if current_c3 < best_c3:
                    best_c3 = current_c3
                    best_step = current_step.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_step[idxs] = old_vals
                no_improve += 1
        
        T = max(T_min, T * alpha)
        
        if no_improve >= patience:
            break
    
    return best_step, best_c3

def _greedy_polish(step, c3, deadline):
    """Greedy single-flip polish to find local optimum."""
    current_step = step.copy()
    current_c3 = c3
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for idx in range(N):
            if time.time() > deadline - 2.0:
                break
            old_val = current_step[idx]
            current_step[idx] = -old_val
            new_c3 = _compute_c3(current_step)
            if new_c3 < current_c3:
                current_c3 = new_c3
                improved = True
            else:
                current_step[idx] = old_val
    return current_step, current_c3

def construct():
    start_time = time.time()
    
    # Phase 1: Start from all-ones (the natural seed) and run SA
    initial = np.ones(N, dtype=np.int8)
    step1, c3_1 = _run_sa(initial, start_time, DEADLINE, seed=42)
    
    # Polish
    if time.time() < DEADLINE - 10.0:
        step1, c3_1 = _greedy_polish(step1, c3_1, DEADLINE)
    
    best_step = step1
    best_c3 = c3_1
    
    # Phase 2: Try a few random restarts with different seeds
    # Only if we have time remaining
    for restart_seed in [123, 456, 789]:
        if time.time() > DEADLINE - 15.0:
            break
        
        # Random initialization with slight bias toward +1
        # This helps because all-ones is a good starting point
        r = np.random.RandomState(restart_seed)
        init = r.choice([-1, 1], size=N)
        # Bias: keep 60% as +1
        bias_mask = r.random(N) < 0.6
        init[bias_mask] = 1
        
        step_r, c3_r = _run_sa(init, start_time, DEADLINE, seed=restart_seed + 1000)
        
        if time.time() < DEADLINE - 5.0:
            step_r, c3_r = _greedy_polish(step_r, c3_r, DEADLINE)
        
        if c3_r < best_c3:
            best_step = step_r
            best_c3 = c3_r
    
    return best_step
