import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def softplus(x):
    # Numerically stable softplus
    return np.where(x > 30, x, np.log1p(np.exp(x)))

def fourier_softplus_to_h(params):
    """
    h(x) = softplus( sum_{k=0}^K a_k cos(2*pi*k*x) )
    params = [a0, a1, ..., aK]
    """
    K = len(params) - 1
    h_raw = np.zeros(N)
    for k in range(K + 1):
        h_raw += params[k] * np.cos(2.0 * np.pi * k * X)
    h = softplus(h_raw)
    return h

def objective(params):
    h = fourier_softplus_to_h(params)
    r = ratio_from_h(h)
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    rng = np.random.default_rng(42)
    
    # Multi-scale optimization
    # Start with low K, increase K
    K_list = [5, 10, 20, 30]
    
    current_params = None
    
    for K in K_list:
        if time.time() > deadline - 20.0:
            break
            
        n_params = K + 1
        bounds = [(-5.0, 5.0)] * n_params
        
        # If we have previous params, use them as initial guess
        if current_params is not None and len(current_params) < n_params:
            x0 = np.zeros(n_params)
            x0[:len(current_params)] = current_params
        else:
            # Random restarts
            n_restarts = 20
            best_params_in_stage = None
            best_val_in_stage = float('inf')
            
            for i in range(n_restarts):
                if time.time() > deadline - 10.0:
                    break
                x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
                # Bias towards positive a0 to ensure h is not too small
                x0[0] = abs(x0[0]) + 0.5
                
                try:
                    result = opt.minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7}
                    )
                    
                    val = result.fun
                    if val < best_val_in_stage:
                        best_val_in_stage = val
                        best_params_in_stage = result.x
                        
                    h_opt = fourier_softplus_to_h(result.x)
                    true_val = ratio_from_h(h_opt)
                    update_best(h_opt, true_val)
                    
                except Exception:
                    pass
            
            current_params = best_params_in_stage
            if current_params is None:
                continue
                
        # Polish the best params from this stage with a few more iterations
        if current_params is not None:
            try:
                result = opt.minimize(
                    objective,
                    current_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_params = result.x
                h_opt = fourier_softplus_to_h(current_params)
                true_val = ratio_from_h(h_opt)
                update_best(h_opt, true_val)
            except Exception:
                pass

    return list(best_h)
