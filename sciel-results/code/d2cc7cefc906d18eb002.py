import numpy as np
from scipy.optimize import minimize
import time

N = 400
DEADLINE = 110.0  # Safety margin before 120s

def _compute_peak(h):
    """Compute max absolute value of autoconvolution."""
    # h is 1D array of length N
    # Linear convolution length is 2N-1
    # Using FFT for efficiency
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    # C has length n_fft, but linear conv is in first 2N-1
    # The rest are zeros (numerically)
    return np.max(np.abs(C[:2*N-1]))

def _compute_raw_score(h):
    """Compute the raw upper bound value: max|conv| / sum(h^2)."""
    peak = _compute_peak(h)
    energy = np.sum(h**2)
    if energy < 1e-8:
        return np.inf
    return peak / energy

def _local_search_discrete(h_init, time_limit):
    """
    Perform local search on discrete +/-1 sequence.
    Returns best h found.
    """
    h = h_init.copy()
    best_h = h.copy()
    best_score = _compute_raw_score(h)
    
    start_time = time.time()
    no_improve_count = 0
    patience = 2000
    
    # Precompute FFT for faster updates? 
    # For N=400, direct convolution might be competitive with FFT update overhead in Python.
    # Let's stick to _compute_peak which uses FFT.
    
    while time.time() - start_time < time_limit:
        # Check deadline globally
        if time.time() > DEADLINE - 5.0:
            break
        
        idx = np.random.randint(N)
        h[idx] = -h[idx]
        
        new_score = _compute_raw_score(h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            no_improve_count = 0
        else:
            h[idx] = -h[idx] # Revert
            no_improve_count += 1
            
        if no_improve_count >= patience:
            # Try a random restart from a perturbed version of best_h
            if time.time() > DEADLINE - 10.0:
                break
            # Perturb 5% of bits
            perturb_idx = np.random.choice(N, size=int(0.05*N), replace=False)
            h[perturb_idx] = -h[perturb_idx]
            no_improve_count = 0
            # Reset patience to allow more search
            patience = 1000

    return best_h

def construct():
    start_time = time.time()
    
    # Phase 1: Continuous Optimization
    # We want to minimize max|conv| / sum(h^2)
    # Use tanh to keep h in (-1, 1)
    
    def objective(x):
        h = np.tanh(x)
        # Compute autoconvolution
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        C = np.fft.irfft(H * H, n=n_fft)
        C = C[:2*N-1]
        
        abs_C = np.abs(C)
        
        # Soft max
        T = 2.0
        max_abs = np.max(abs_C)
        # Stable log-sum-exp
        shifted = (abs_C - max_abs) / T
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
        soft_max = T * log_sum_exp
        
        sum_sq = np.sum(h**2)
        
        # Avoid division by zero
        if sum_sq < 1e-8:
            return 1e10
            
        return soft_max / sum_sq

    # Initial guess: All ones (x=5)
    x0 = np.full(N, 5.0)
    
    # Alternative initial guesses
    # Alternating
    x0_alt = np.zeros(N)
    x0_alt[::2] = 2.0
    x0_alt[1::2] = -2.0
    
    best_x = None
    best_val = np.inf
    
    # Run optimization for x0
    if time.time() - start_time < DEADLINE - 60.0:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    # Run optimization for x0_alt if time permits
    if time.time() - start_time < DEADLINE - 30.0:
        res_alt = minimize(objective, x0_alt, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
        if res_alt.fun < best_val:
            best_val = res_alt.fun
            best_x = res_alt.x

    # If no optimization was run or failed, fallback to all ones
    if best_x is None:
        h_cont = np.ones(N)
    else:
        h_cont = np.tanh(best_x)
        
    # Phase 2: Discretization
    # Threshold to +/-1
    h_disc = np.sign(h_cont)
    # Ensure no zeros (sign(0)=0, replace with 1)
    h_disc[h_disc == 0] = 1
    
    # Phase 3: Local Search
    # Allocate remaining time to local search
    elapsed = time.time() - start_time
    time_for_ls = DEADLINE - elapsed - 5.0 # 5s safety margin
    
    if time_for_ls > 5.0:
        h_best = _local_search_discrete(h_disc, time_for_ls)
    else:
        h_best = h_disc
        
    return h_best
