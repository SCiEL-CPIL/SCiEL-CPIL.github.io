import numpy as np
import time

N = 400
DEADLINE = 118.0
M = 4 * N  # fine grid size
dx = 0.5 / M  # domain length is 0.5
REP = M // N  # repeat factor

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the step function.
    Uses FFT on a fine grid for efficiency.
    """
    f_fine = np.repeat(signed_step.astype(np.float64), REP)
    fft_f = np.fft.rfft(f_fine, n=2 * M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
    autoconv *= dx
    return np.max(np.abs(autoconv))

def _autoconv_from_fine(f_fine):
    """Compute autoconvolution from fine grid representation."""
    fft_f = np.fft.rfft(f_fine, n=2 * M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
    autoconv *= dx
    return autoconv

def _smooth_max_and_grad(f, beta=50.0):
    """
    Compute smooth max |autoconv| and its gradient w.r.t. f (coarse grid).
    f: shape (N,) continuous values in [-1, 1]
    Returns: (smooth_max_value, gradient shape (N,))
    """
    f_fine = np.repeat(f, REP)
    autoconv = _autoconv_from_fine(f_fine)
    abs_ac = np.abs(autoconv)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-12:
        return 0.0, np.zeros(N)
    
    # Soft-max weights
    shifted = abs_ac / max_val
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Smooth max approximation
    log_sum = np.log(np.sum(weights))
    smooth_max = max_val * (1.0 + log_sum / beta)
    
    # Gradient: use subgradient of max-norm (dominated by max component)
    k = np.argmax(abs_ac)
    n = 2 * M
    sign_c = np.sign(autoconv[k])
    
    # grad_fine[m] = sign_c * (f_fine[(k-m) mod n] + f_fine[(k+m) mod n])
    # Vectorized computation
    idx1 = (k - np.arange(n)) % n
    idx2 = (k + np.arange(n)) % n
    grad_fine = sign_c * (f_fine[idx1] + f_fine[idx2])
    
    # Aggregate to coarse grid
    grad_coarse = np.zeros(N)
    for j in range(N):
        start = j * REP
        end = start + REP
        grad_coarse[j] = np.sum(grad_fine[start:end])
    
    return smooth_max, grad_coarse

def _projected_gradient_descent(f_init, deadline, max_iter=1500, lr_init=0.005):
    """
    Projected gradient descent on the smooth surrogate.
    """
    f = f_init.copy().astype(np.float64)
    lr = lr_init
    best_f = f.copy()
    best_obj = np.inf
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        obj, grad = _smooth_max_and_grad(f)
        
        if obj < best_obj:
            best_obj = obj
            best_f = f.copy()
        
        # Gradient step
        f_new = f - lr * grad
        # Project to [-1, 1]^N
        f_new = np.clip(f_new, -1.0, 1.0)
        
        f = f_new
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.9
    
    return best_f

def _discretize(f):
    """Discretize continuous values to ±1."""
    return np.sign(f).astype(np.int8)

def _run_sa(initial_step, start_time, deadline, seed):
    """Run simulated annealing from an initial configuration."""
    np.random.seed(seed)
    
    current_step = initial_step.copy()
    current_c3 = _compute_c3(current_step)
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # SA parameters
    T = 0.01  # initial temperature
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
    
    # Phase 1: Continuous optimization to find a good initial point
    f_init = np.ones(N, dtype=np.float64)
    
    # Run projected gradient descent
    f_cont = _projected_gradient_descent(f_init, DEADLINE, max_iter=1500, lr_init=0.005)
    
    # Discretize
    step_cont = _discretize(f_cont)
    c3_cont = _compute_c3(step_cont)
    
    # Polish the discretized result
    if time.time() < DEADLINE - 10.0:
        step_cont, c3_cont = _greedy_polish(step_cont, c3_cont, DEADLINE)
    
    best_step = step_cont
    best_c3 = c3_cont
    
    # Phase 2: Also try from all-ones with SA (original approach)
    initial = np.ones(N, dtype=np.int8)
    if time.time() < DEADLINE - 15.0:
        step_sa, c3_sa = _run_sa(initial, start_time, DEADLINE, seed=42)
        if time.time() < DEADLINE - 5.0:
            step_sa, c3_sa = _greedy_polish(step_sa, c3_sa, DEADLINE)
        if c3_sa < best_c3:
            best_step = step_sa
            best_c3 = c3_sa
    
    # Phase 3: Try random restarts if time permits
    for restart_seed in [123, 456]:
        if time.time() > DEADLINE - 15.0:
            break
        
        r = np.random.RandomState(restart_seed)
        init = r.choice([-1, 1], size=N)
        bias_mask = r.random(N) < 0.6
        init[bias_mask] = 1
        
        step_r, c3_r = _run_sa(init, start_time, DEADLINE, seed=restart_seed + 1000)
        
        if time.time() < DEADLINE - 5.0:
            step_r, c3_r = _greedy_polish(step_r, c3_r, DEADLINE)
        
        if c3_r < best_c3:
            best_step = step_r
            best_c3 = c3_r
    
    return best_step
