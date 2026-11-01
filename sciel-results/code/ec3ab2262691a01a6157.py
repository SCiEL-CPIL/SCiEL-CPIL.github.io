"""Combined program: continuous spectral shaping + discrete local search for C3 minimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 118.0  # Safety margin before 120s limit


def _compute_c3(h):
    """
    Compute the C3 upper bound for a signed step function with N equal intervals on [-1/4, 1/4].
    The step function f(x) = h[i] for x in [-1/4 + i/N, -1/4 + (i+1)/N).
    C3 = max_t |(f*f)(t)| where (f*f)(t) = integral f(s) f(t-s) ds.
    
    For efficiency, we compute the autoconvolution via FFT on a fine grid.
    """
    # Use a fine grid for accurate autoconvolution
    M = 4 * N  # 4x oversampling
    dx = 0.5 / M  # domain length is 0.5
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = h[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # C3 = max |autoconv| over the full range
    c3 = np.max(np.abs(autoconv))
    
    return c3


def _compute_autocorr_direct(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N (for the discrete sequence).
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac


def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max |ac[k]| for k in [1, N-1] using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
    
    shifted = beta * (sidelobes - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    weights = beta * exp_shifted / sum_exp
    return obj, weights


def _objective_and_gradient(x, beta=50.0):
    """
    Objective: soft max of |autocorrelation sidelobes| + penalty on non-binary values.
    """
    h = x
    ac = _compute_autocorr_direct(h)
    sidelobes = ac[1:N]
    
    sidelobes_abs = np.abs(sidelobes)
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
    
    # Stable log-sum-exp
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient computation
    weights = beta * exp_shifted / sum_exp
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k - 1]
        if w_k < 1e-15:
            continue
        
        sign_k = np.sign(sidelobes[k - 1])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        grad[0:N - k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N - k]
    
    # Add penalty to encourage binary values
    penalty_grad = 0.1 * (h - np.sign(h))
    grad += penalty_grad
    obj += 0.05 * np.sum(h**2 - 1)
    
    return obj, grad


def _local_search_fast(h, max_iterations=10000, time_limit=5.0, use_sa=False, T0=0.01, c3_mode=True):
    """
    Fast local search with incremental autocorrelation update.
    If c3_mode, we optimize the full C3 metric; otherwise, just sidelobes.
    """
    start_time = time.time()
    
    if c3_mode:
        # For C3 mode, we need to track the full autoconvolution
        # But for speed, we'll use the discrete autocorrelation as a proxy
        # and verify with C3 at the end
        ac = _compute_autocorr_direct(h)
        # Use max of full autocorrelation as proxy
        current_val = np.max(np.abs(ac))
    else:
        ac = _compute_autocorr_direct(h)
        current_val = np.max(np.abs(ac[1:N]))
    
    best_val = current_val
    best_h = h.copy()
    
    if use_sa and current_val > 1e-10:
        T = T0
    else:
        T = 0.0
    
    no_improve = 0
    patience = 500
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        i = np.random.randint(N)
        old_h_i = h[i]
        new_h_i = -old_h_i
        
        # Incremental update of autocorrelation
        delta_ac = np.zeros(2 * N)
        
        for k in range(1, N):
            change = 0.0
            if i + k < N:
                change += h[i + k]
            if i - k >= 0:
                change += h[i - k]
            delta_ac[k] = -change
            delta_ac[2 * N - k] = -change
        
        ac_new = ac + delta_ac
        
        if c3_mode:
            new_val = np.max(np.abs(ac_new))
        else:
            new_val = np.max(np.abs(ac_new[1:N]))
        
        if new_val < current_val:
            h[i] = new_h_i
            ac = ac_new
            current_val = new_val
            no_improve = 0
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            accepted = False
            if T > 1e-9:
                delta = new_val - current_val
                if delta < 0:
                    accepted = True
                else:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        accepted = True
            
            if accepted:
                h[i] = new_h_i
                ac = ac_new
                current_val = new_val
            else:
                h[i] = old_h_i
                no_improve += 1
        
        if T > 1e-9:
            T *= 0.9995
        
        if no_improve >= patience:
            break
    
    return best_h, best_val


def _verify_c3(h):
    """Verify the actual C3 value using the fine-grid method."""
    return _compute_c3(h)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_c3 = float('inf')
    
    # Phase 1: Continuous Relaxation with Multiple Strategies
    num_continuous_starts = 6
    betas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 30.0:
            break
            
        rng = np.random.default_rng(42 + i * 137)
        
        # Different initialization strategies
        if i == 0:
            # All ones (seed)
            x0 = np.ones(N)
        elif i == 1:
            # Random binary
            x0 = rng.choice([-1.0, 1.0], size=N)
        elif i == 2:
            # Near-binary with noise
            x0 = rng.choice([-1.0, 1.0], size=N) * 0.9 + rng.normal(0, 0.1, size=N)
            x0 = np.clip(x0, -1.0, 1.0)
        elif i == 3:
            # Continuous random
            x0 = rng.uniform(-1.0, 1.0, size=N)
        else:
            # Start from best found so far with perturbation
            if best_h is not None:
                x0 = best_h.copy().astype(float)
                mask = rng.random(N) < 0.2
                x0[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            else:
                x0 = rng.choice([-1.0, 1.0], size=N)
        
        beta = betas[i % len(betas)]
        
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * N,
                           options={'maxiter': 20, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 10, 'disp': False})
        except Exception:
            continue
        
        # Round to binary using multiple strategies
        candidates = []
        
        # Strategy 1: sign
        h_bin1 = np.sign(res.x)
        zero_mask = (h_bin1 == 0)
        if np.any(zero_mask):
            h_bin1[zero_mask] = rng.choice([-1.0, 1.0], size=np.sum(zero_mask))
        candidates.append(h_bin1.astype(np.int8))
        
        # Strategy 2: threshold at 0 with random tie-breaking
        h_bin2 = np.where(res.x > 0, 1.0, -1.0).astype(np.int8)
        candidates.append(h_bin2)
        
        # Strategy 3: probabilistic rounding
        probs = 0.5 * (1 + res.x)
        h_bin3 = np.where(rng.random(N) < probs, 1.0, -1.0).astype(np.int8)
        candidates.append(h_bin3)
        
        for h_bin in candidates:
            # Verify with actual C3
            c3_val = _verify_c3(h_bin)
            
            if c3_val < best_c3:
                best_c3 = c3_val
                best_h = h_bin.copy()
            
            # Quick local search refinement (using discrete autocorr as proxy for speed)
            if time.time() < deadline - 20.0:
                h_ref, val_ref = _local_search_fast(h_bin.copy(), max_iterations=300, time_limit=0.3, c3_mode=False)
                # Verify the refined solution with actual C3
                c3_ref = _verify_c3(h_ref)
                if c3_ref < best_c3:
                    best_c3 = c3_ref
                    best_h = h_ref.copy()

    # Phase 2: Thorough local search on best solution
    if best_h is not None and time.time() < deadline - 10.0:
        h_ref, val_ref = _local_search_fast(best_h.copy(), max_iterations=10000, time_limit=8.0, c3_mode=False)
        c3_ref = _verify_c3(h_ref)
        if c3_ref < best_c3:
            best_c3 = c3_ref
            best_h = h_ref.copy()

    # Phase 3: Final SA polish
    remaining_time = deadline - time.time()
    if remaining_time > 3.0 and best_h is not None:
        T0 = max(0.005, best_c3 * 0.02)
        h_final, val_final = _local_search_fast(best_h.copy(), max_iterations=30000, time_limit=remaining_time - 1.0, use_sa=True, T0=T0, c3_mode=False)
        c3_final = _verify_c3(h_final)
        if c3_final < best_c3:
            best_c3 = c3_final
            best_h = h_final.copy()

    if best_h is None:
        return np.ones(N, dtype=np.int8)
        
    return best_h
