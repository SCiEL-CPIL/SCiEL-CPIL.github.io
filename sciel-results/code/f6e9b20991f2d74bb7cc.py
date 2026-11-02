"""
Optimized program for signed step functions minimizing max absolute autoconvolution 
divided by squared integral.

Strategy:
1. Use a smooth surrogate objective for the max absolute autoconvolution.
2. Optimize the phases of the frequency domain representation (PSD) using gradient descent.
   This allows for global moves that reduce the peak autoconvolution.
3. Quantize the resulting continuous signal to signed step heights (±1).
4. Perform a final local search refinement on the discrete sequence.
"""

import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_objective_and_grad(phases, beta=50.0):
    """
    Compute the smooth objective and gradient with respect to phases.
    phases: array of length N (real signal, so we use N phases, symmetry handled by IFFT)
    Actually, for a real signal x, X[k] = X[N-k]^*. 
    We can just optimize N phases and take the real part of the IFFT.
    """
    # Construct complex spectrum
    X = np.exp(1j * phases)
    
    # Inverse FFT to get time domain signal
    x = np.fft.irfft(X, n=N)
    
    # Calculate autoconvolution
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X_full = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X_full * X_full, n=L_fft)[:L]
    
    # Objective: max |C| / (sum x)^2
    # Smooth max: (1/beta) * log(sum(exp(beta * |C|)))
    # We want to minimize this.
    # Note: |C| is non-negative.
    
    sum_x = np.sum(x)
    if abs(sum_x) < 1e-6:
        # Penalize zero sum
        return 1e10, np.zeros_like(phases)
    
    # Smooth max of |C|
    # To avoid overflow, subtract max
    C_abs = np.abs(C)
    max_C = np.max(C_abs)
    if max_C == 0:
        smooth_max = 0.0
    else:
        # log-sum-exp trick
        shifted = beta * (C_abs - max_C)
        smooth_max = max_C + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    
    obj = smooth_max / (sum_x * sum_x)
    
    # Gradient calculation
    # d obj / d phases
    # This is complex. We use the chain rule.
    # obj = f(C, sum_x)
    # d obj / d x = df/dC * dC/dx + df/dsum * dsum/dx
    # dC/dx is linear (convolution).
    
    # Simplified gradient estimation or analytical?
    # Analytical:
    # Let M = smooth_max.
    # dM/dC_i = (beta * exp(beta * |C_i|)) / sum(exp(beta * |C_j|)) * sign(C_i)
    # Actually, d/dC_i (1/beta log sum exp(beta |C_j|)) = exp(beta |C_i|) / sum exp(beta |C_j|) * sign(C_i)
    
    weights = np.exp(beta * (C_abs - max_C))
    weights_sum = np.sum(weights)
    dM_dC = (weights / weights_sum) * np.sign(C)
    
    # d obj / d C = dM_dC / (sum_x^2)
    d_obj_dC = dM_dC / (sum_x * sum_x)
    
    # C = x * x
    # d C_m / d x_i = x_{m-i} + x_{m+i} (with appropriate indexing/boundaries)
    # In frequency domain, it's easier.
    # C = IFFT( |FFT(x)|^2 )
    # This is getting complicated. Let's use numerical gradient if analytical is too slow/risky?
    # No, numerical gradient for N=400 is 400 evaluations of O(N log N). 400 * 400 log 400 ~ 1.6M ops. Feasible.
    # But we want speed.
    
    # Let's use the property:
    # grad_x = 2 * IFFT( FFT(x) * conj(FFT(d_obj_dC)) )? No.
    # C = x * x.
    # <grad_C, C> = <grad_x, 2 * (x * grad_C)?> No.
    # The gradient of the autoconvolution with respect to x is 2 * (x * g) where g is the gradient of the cost w.r.t C.
    # Specifically, d/dx_i (sum_m C_m g_m) = 2 sum_m x_{m-i} g_m = 2 (x * g)_i.
    
    # So d_obj_d_x = 2 * ifft( fft(x) * fft(d_obj_dC) )? 
    # Wait, C is linear in x? No, C is quadratic.
    # d/dx_i (C * g) = 2 * (x * g)_i.
    
    # We need d_obj_d_x.
    # d_obj/d_x_i = 2 * sum_m d_obj_dC_m * (d C_m / d x_i)
    # d C_m / d x_i = x_{m-i} + x_{m+i} (if indices valid)
    # This is equivalent to 2 * (x * d_obj_dC)_i in the circular convolution sense, 
    # but our autoconv is linear.
    
    # Let's just use numerical gradient for the phases? 
    # N=400. 400 perturbations.
    # Each perturbation requires an IFFT and FFT.
    # 400 * (400 log 400) ~ 1.6M flops.
    # In Python, this might be slow if not vectorized.
    
    # Alternative: Just use L-BFGS-B from scipy on the phases, providing the objective function.
    # scipy.optimize.minimize will handle the gradient estimation or we can provide it.
    # Since we don't have a clean analytical gradient for the phase domain directly 
    # without the x->C->obj chain, and the chain is complex, 
    # let's try a simple coordinate descent on phases? No, too slow.
    
    # Let's try a different structural approach:
    # Random restarts of local search on ±1 sequences, but with a better initialization 
    # derived from the continuous relaxation.
    
    # 1. Optimize phases using scipy L-BFGS-B on the smooth objective.
    # 2. Quantize.
    # 3. Local search.
    
    return obj, None # Placeholder, we will use scipy's numerical gradient if needed, or implement properly.

def _construct_continuous_optimal(N, max_iter=1000, beta=50.0):
    """
    Use scipy.optimize.minimize to find phases that minimize the smooth objective.
    """
    from scipy.optimize import minimize
    
    def obj_and_grad(phases):
        # Compute x
        X = np.exp(1j * phases)
        x = np.fft.irfft(X, n=N)
        
        # Compute C
        L = 2 * N - 1
        L_fft = 1
        while L_fft < L:
            L_fft *= 2
        X_full = np.fft.rfft(x, n=L_fft)
        C = np.fft.irfft(X_full * X_full, n=L_fft)[:L]
        
        sum_x = np.sum(x)
        if abs(sum_x) < 1e-6:
            return 1e10, np.zeros_like(phases)
        
        C_abs = np.abs(C)
        max_C = np.max(C_abs)
        if max_C == 0:
            smooth_max = 0.0
            dM_dC = np.zeros_like(C)
        else:
            shifted = beta * (C_abs - max_C)
            exp_shifted = np.exp(shifted)
            smooth_max = max_C + (1.0 / beta) * np.log(np.sum(exp_shifted))
            dM_dC = (exp_shifted / np.sum(exp_shifted)) * np.sign(C)
            
        obj = smooth_max / (sum_x * sum_x)
        
        # Gradient w.r.t x
        # d_obj_dC = dM_dC / (sum_x^2)
        d_obj_dC = dM_dC / (sum_x * sum_x)
        
        # d_obj_d_x = 2 * (x * d_obj_dC)
        # Linear convolution.
        # We can compute this using FFT.
        # conv(x, d_obj_dC)
        L_conv = N + len(d_obj_dC) - 1
        L_fft_conv = 1
        while L_fft_conv < L_conv:
            L_fft_conv *= 2
        
        X_x = np.fft.rfft(x, n=L_fft_conv)
        X_g = np.fft.rfft(d_obj_dC, n=L_fft_conv)
        conv = np.fft.irfft(X_x * X_g, n=L_fft_conv)
        
        # The gradient of the quadratic form x*x is 2 * (x * g)
        d_obj_d_x = 2 * conv[:N] # We only care about the first N components?
        # Wait, the gradient w.r.t x_i is sum_m dC_m/dx_i * d_obj_dC_m.
        # dC_m/dx_i = x_{m-i} + x_{m+i}.
        # This is exactly the linear convolution of x and g, plus x and reversed g?
        # Autoconvolution is x * x.
        # Gradient is 2 * (x * g).
        
        # Now map gradient from x to phases.
        # x = IFFT(exp(i * phases))
        # dx/dphase_k = i * X_k * (1/N) * ...?
        # x_n = (1/N) sum_k X_k exp(i 2 pi n k / N)
        # X_k = exp(i phi_k)
        # dx_n/dphi_k = (1/N) i X_k exp(i 2 pi n k / N)
        # This is complex.
        # d_obj/dphi_k = sum_n (d_obj/dx_n) * (dx_n/dphi_k)
        # = sum_n d_obj_d_x_n * (1/N) i X_k exp(i 2 pi n k / N)
        # = (i X_k / N) sum_n d_obj_d_x_n exp(i 2 pi n k / N)
        # = (i X_k / N) * DFT(d_obj_d_x)_k
        
        DFT_dx = np.fft.rfft(d_obj_d_x, n=N)
        grad_phases = np.real(-1j * X * DFT_dx / N) # -i * i = 1? 
        # dx/dphi = i X ...
        # d_obj/dphi = <d_obj/dx, dx/dphi>
        # = sum_n d_obj_d_x_n * (i/N) X_k exp(...)
        # = (i X_k / N) DFT(d_obj_d_x)_k
        # We want the real gradient.
        
        return obj, grad_phases

    # Initialize with random phases
    rng = np.random.default_rng(42)
    x0 = rng.uniform(-np.pi, np.pi, N)
    
    try:
        res = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        best_phases = res.x
    except Exception:
        best_phases = x0
        
    # Convert best phases to x
    X = np.exp(1j * best_phases)
    x_cont = np.fft.irfft(X, n=N)
    
    # Quantize to ±1
    x_quant = np.sign(x_cont)
    # Handle zeros
    x_quant[x_quant == 0] = 1.0
    
    return x_quant

def _local_search(x, max_iter=50000, seed=42):
    """
    Fast local search on ±1 sequences.
    """
    rng = np.random.default_rng(seed)
    current_val = _objective(x)
    best_x = x.copy()
    best_val = current_val
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
        
    for _ in range(max_iter):
        if _check_deadline():
            break
            
        pos = rng.integers(0, N)
        old_val = x[pos]
        x[pos] = -old_val
        
        new_val = _objective(x)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[pos] = old_val
            
    return best_x

def construct():
    # Strategy 1: Try the trivial all-ones sequence
    x_ones = np.ones(N)
    val_ones = _objective(x_ones)
    
    # Strategy 2: Continuous optimization
    x_cont = _construct_continuous_optimal(N, max_iter=200)
    val_cont = _objective(x_cont)
    
    best_x = x_ones if val_ones < val_cont else x_cont
    best_val = min(val_ones, val_cont)
    
    if _check_deadline():
        return best_x
        
    # Strategy 3: Local search refinement
    x_refined = _local_search(best_x, max_iter=20000, seed=123)
    val_refined = _objective(x_refined)
    
    if val_refined < best_val:
        best_x = x_refined
        best_val = val_refined
        
    if _check_deadline():
        return best_x
        
    # Strategy 4: Another round of local search with different seed
    x_refined2 = _local_search(best_x, max_iter=20000, seed=456)
    val_refined2 = _objective(x_refined2)
    
    if val_refined2 < best_val:
        best_x = x_refined2
        best_val = val_refined2
        
    return best_x
# EVOLVE-BLOCK-END
