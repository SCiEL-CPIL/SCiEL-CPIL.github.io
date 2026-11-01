import numpy as np
from scipy.optimize import minimize
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_peak(h):
    """Compute max absolute value of autoconvolution."""
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
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
    
    while time.time() - start_time < time_limit:
        # Check deadline globally
        if time.time() > DEADLINE - 2.0:
            break
        
        # Try flipping a random subset of indices for more effective search
        # Sometimes flip multiple bits to escape local minima
        if np.random.random() < 0.1:
            # Flip 2-5 random bits
            k = np.random.randint(2, 6)
            idxs = np.random.choice(N, size=k, replace=False)
            old_vals = h[idxs].copy()
            h[idxs] = -h[idxs]
        else:
            # Single bit flip
            idx = np.random.randint(N)
            idxs = np.array([idx])
            old_vals = h[idxs].copy()
            h[idxs] = -h[idxs]
        
        new_score = _compute_raw_score(h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            no_improve_count = 0
        else:
            h[idxs] = old_vals  # Revert
            no_improve_count += 1
            
        if no_improve_count >= patience:
            # Try a random restart from a perturbed version of best_h
            if time.time() > DEADLINE - 5.0:
                break
            # Perturb 5-10% of bits
            frac = np.random.uniform(0.05, 0.1)
            perturb_idx = np.random.choice(N, size=int(frac*N), replace=False)
            h[perturb_idx] = -h[perturb_idx]
            no_improve_count = 0
            # Reset patience
            patience = 1000

    return best_h

def construct():
    start_time = time.time()
    
    # Phase 1: Continuous Optimization using random-walk parameterization
    # Represent h as differences of a random walk to encourage smoothness
    # h[i] = s[i+1] - s[i] where s is the cumulative sum
    # We optimize s directly, then take h = diff(s)
    
    def objective(x):
        # x represents the cumulative sum s
        # h is the difference: h[i] = s[i+1] - s[i]
        # But we need h to be +/-1, so we use tanh to keep in (-1,1)
        s = x
        h = np.tanh(np.diff(s, prepend=s[0]-s[0]))  # This gives 0, so let's rethink
        
        # Better: let x be the heights directly, but add a smoothness penalty
        # to encourage low-variance spectral profile
        h = np.tanh(x)
        
        # Compute autoconvolution
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        C = np.fft.irfft(H * H, n=n_fft)
        C = C[:2*N-1]
        
        abs_C = np.abs(C)
        
        # Soft max with temperature
        T = 1.0
        max_abs = np.max(abs_C)
        shifted = (abs_C - max_abs) / T
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
        soft_max = T * log_sum_exp
        
        sum_sq = np.sum(h**2)
        
        if sum_sq < 1e-8:
            return 1e10
        
        # Add a smoothness penalty to encourage low second-order differences
        # This corresponds to the random-walk idea: penalize rapid changes
        if len(h) > 2:
            second_diff = np.diff(h, n=2)
            smoothness_penalty = 0.01 * np.sum(second_diff**2)
        else:
            smoothness_penalty = 0.0
            
        return (soft_max + smoothness_penalty) / sum_sq

    # Initial guess: All ones
    x0 = np.full(N, 5.0)
    
    # Alternative initial guesses
    x0_alt = np.zeros(N)
    x0_alt[::2] = 2.0
    x0_alt[1::2] = -2.0
    
    # Another: random
    np.random.seed(42)
    x0_rand = np.random.randn(N) * 2.0
    
    best_x = None
    best_val = np.inf
    
    # Run optimization for x0
    if time.time() - start_time < DEADLINE - 60.0:
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Run optimization for x0_alt if time permits
    if time.time() - start_time < DEADLINE - 30.0:
        try:
            res_alt = minimize(objective, x0_alt, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-8})
            if res_alt.fun < best_val:
                best_val = res_alt.fun
                best_x = res_alt.x
        except Exception:
            pass

    # Run optimization for x0_rand if time permits
    if time.time() - start_time < DEADLINE - 15.0:
        try:
            res_rand = minimize(objective, x0_rand, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            if res_rand.fun < best_val:
                best_val = res_rand.fun
                best_x = res_rand.x
        except Exception:
            pass

    # If no optimization was run or failed, fallback to all ones
    if best_x is None:
        h_cont = np.ones(N)
    else:
        h_cont = np.tanh(best_x)
        
    # Phase 2: Discretization
    h_disc = np.sign(h_cont)
    h_disc[h_disc == 0] = 1
    
    # Phase 3: Local Search
    elapsed = time.time() - start_time
    time_for_ls = DEADLINE - elapsed - 3.0
    
    if time_for_ls > 5.0:
        h_best = _local_search_discrete(h_disc, time_for_ls)
    else:
        h_best = h_disc
        
    # Final check: ensure we return a valid +/-1 array
    h_best = np.sign(h_best)
    h_best[h_best == 0] = 1
    
    return h_best
