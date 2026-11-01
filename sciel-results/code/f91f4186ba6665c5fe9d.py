"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_acf(h):
    """Compute linear autocorrelation using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf

def _compute_c3_bound_from_acf(acf):
    """Compute C3 bound from precomputed autocorrelation."""
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3) / (N ** 3)

def _compute_c3_bound(h):
    """Compute C3 bound from sequence."""
    acf = _compute_acf(h)
    return _compute_c3_bound_from_acf(acf)

def _continuous_objective(x, n_fft_cache):
    """
    Continuous objective: minimize sum of cubes of sidelobes of autocorrelation.
    x is a real vector of length N (continuous relaxation of binary sequence).
    """
    n_fft = n_fft_cache
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3)

def _continuous_gradient(x, n_fft_cache):
    """
    Gradient of the continuous objective w.r.t. x.
    Uses FFT-based differentiation.
    """
    n_fft = n_fft_cache
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    
    # Gradient of sum(|acf[k]|^3) w.r.t. x[i]
    # d/dx_i sum_k |acf[k]|^3 = sum_k 3|acf[k]|^2 * sign(acf[k]) * d(acf[k])/dx_i
    # acf[k] = sum_i x[i] * x[i+k] (for k >= 1, with appropriate boundary handling)
    # Using FFT: acf is the inverse FFT of |F|^2
    # d(acf[k])/dx[i] involves the structure of the autocorrelation
    
    # Simpler: use numerical gradient via FFT
    # Actually, let's use the analytical form:
    # acf[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k] for k >= 1
    # d(acf[k])/dx[i] = x[i+k] (if i+k < N) + x[i-k] (if i-k >= 0)
    
    # This is complex to compute analytically. Let's use a simpler approach:
    # The gradient of sum_k |acf[k]|^3 can be computed as:
    # grad_x = 3 * sum_k |acf[k]|^2 * sign(acf[k]) * (x * shifted pattern)
    
    # Actually, let's just use L-BFGS-B with finite differences for gradient
    # This is slower but more robust for this non-convex problem
    
    return None  # Let scipy use finite differences

def _local_search_binary(h, deadline, rng, max_rounds=50):
    """
    Fast local search on binary sequence: flip bits that improve the objective.
    """
    acf = _compute_acf(h)
    current_val = _compute_c3_bound_from_acf(acf)
    best_h = h.copy()
    best_val = current_val
    
    indices = np.arange(N)
    improved = True
    rounds = 0
    
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                return best_h, best_val
            
            # Try flipping h[i]
            h[i] = -h[i]
            new_acf = _compute_acf(h)
            new_val = _compute_c3_bound_from_acf(new_acf)
            
            if new_val < current_val:
                current_val = new_val
                acf = new_acf
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i] = -h[i]  # Revert
    
    return best_h, best_val

def _refine_binary(h, deadline, rng):
    """
    Refine a binary sequence using local search with restarts.
    """
    acf = _compute_acf(h)
    current_val = _compute_c3_bound_from_acf(acf)
    best_h = h.copy()
    best_val = current_val
    
    # Multiple rounds of local search with random perturbations
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Local search
        res_h, res_val = _local_search_binary(h.copy(), deadline, rng, max_rounds=30)
        
        if res_val < best_val:
            best_val = res_val
            best_h = res_h.copy()
            h = res_h.copy()
        else:
            # Perturb to escape local minimum
            n_flips = rng.integers(1, max(2, N // 20))
            flip_idx = rng.choice(N, size=n_flips, replace=False)
            h = best_h.copy()
            h[flip_idx] = -h[flip_idx]
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = np.inf
    
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Strategy 1: Continuous relaxation via L-BFGS-B
    # Use multiple random initializations in the continuous space
    for restart in range(8):
        if time.time() > deadline - 30:
            break
        
        # Random continuous initialization in [-1, 1]
        x0 = rng.uniform(-1.0, 1.0, size=N)
        
        try:
            result = minimize(
                _continuous_objective,
                x0,
                args=(n_fft,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if result.success or result.fun < best_val:
                # Threshold to binary
                h_bin = np.sign(result.x)
                h_bin[h_bin == 0] = 1.0
                
                acf = _compute_acf(h_bin)
                val = _compute_c3_bound_from_acf(acf)
                
                if val < best_val:
                    best_val = val
                    best_h = h_bin.copy()
                    
        except Exception:
            continue
    
    # Strategy 2: Random binary initialization + local search
    for restart in range(15):
        if time.time() > deadline - 15:
            break
        
        h = rng.choice([-1.0, 1.0], size=N)
        acf = _compute_acf(h)
        val = _compute_c3_bound_from_acf(acf)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Refine
        if time.time() < deadline - 5:
            res_h, res_val = _refine_binary(h, deadline, rng)
            if res_val < best_val:
                best_val = res_val
                best_h = res_h
    
    # Final refinement on best result
    if best_h is not None and time.time() < deadline - 2:
        res_h, res_val = _refine_binary(best_h, deadline, rng)
        if res_val < best_val:
            best_h = res_h
            best_val = res_val
    
    return best_h

# EVOLVE-BLOCK-END
