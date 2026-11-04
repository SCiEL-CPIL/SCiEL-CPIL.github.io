import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
# Using a slightly finer grid than Inspiration 1 (6000) for accuracy, 
# but not as fine as Current Program (12000) to save time for DE.
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Fourier basis for K modes
K = 25
I = np.arange(N)
COS = np.cos(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)
SIN = np.sin(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)

def fourier_to_heights(params):
    """
    Convert Fourier parameters to nonnegative step heights.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    """
    a0 = params[0]
    a_coeffs = params[1::2][:K]
    b_coeffs = params[2::2][:K]
    
    heights = a0 + a_coeffs @ COS + b_coeffs @ SIN
    return np.clip(heights, 0.0, None)

def heights_to_fourier(heights):
    """
    Compute Fourier coefficients of a height vector.
    """
    a0 = np.mean(heights)
    a_coeffs = np.zeros(K)
    b_coeffs = np.zeros(K)
    for k in range(1, K + 1):
        a_coeffs[k-1] = (2.0 / N) * np.sum(heights * np.cos(2.0 * np.pi * k * I / N))
        b_coeffs[k-1] = (2.0 / N) * np.sum(heights * np.sin(2.0 * np.pi * k * I / N))
    
    params = np.zeros(2 * K + 1)
    params[0] = a0
    params[1::2] = a_coeffs
    params[2::2] = b_coeffs
    return params

def ratio_from_h(h):
    """
    Compute max(autoconv) / ||h||_2^2 using FFT on fine grid.
    This matches the Current Program's scoring which yielded 0.888.
    """
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

def smooth_max(values, beta=50.0):
    """Soft-max via log-sum-exp for smooth surrogate of max."""
    m = np.max(values)
    if m < -1000:
        return m
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def smooth_objective_fourier(params, beta=30.0):
    """
    Smooth objective for Fourier parameters.
    Uses L2 normalization (matching Current Program).
    """
    h = fourier_to_heights(params)
    s = h.sum()
    if s < 1e-15:
        return 1e6
    
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return 1e6
    
    # Normalize conv by l2 to make it scale invariant
    conv_scaled = conv / l2
    smax = smooth_max(conv_scaled, beta)
    return smax

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    # Initial best: uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    rng = np.random.default_rng(42)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()

    # Phase 1: Global Optimization using Differential Evolution on Fourier Coefficients
    # This leverages the structural insight from Inspiration 1 but optimizes for the L2 metric
    # that proved effective in the Current Program.
    
    n_params = 2 * K + 1
    bounds = [(0.0, 5.0)] * n_params  # a0 and coefficients
    
    # We want a0 to be positive. Coefficients can be negative.
    # Let's define bounds more carefully.
    # a0 should be around 1.0. Coeffs can be -2 to 2.
    bounds[0] = (0.5, 3.0)
    for i in range(1, n_params):
        bounds[i] = (-2.0, 2.0)

    try:
        de_result = opt.differential_evolution(
            smooth_objective_fourier,
            bounds,
            args=(20.0,), # beta for smooth max
            seed=42,
            maxiter=50,
            tol=1e-4,
            polish=False, # We will polish manually
            update_interval=1,
            callback=lambda x, convergence: None
        )
        
        if de_result.success or np.isfinite(de_result.fun):
            h_de = fourier_to_heights(de_result.x)
            val_de = ratio_from_h(h_de)
            update_best(h_de, val_de)
    except Exception as e:
        pass

    # Phase 2: Local Refinement using L-BFGS-B on Fourier Coefficients
    # Start from the best Fourier result found so far (or uniform if DE failed)
    if time.time() < deadline - 30.0:
        # Convert current best_h to Fourier params
        p_start = heights_to_fourier(best_h)
        
        # Try a few restarts from slightly perturbed Fourier params
        for i in range(5):
            if time.time() > deadline - 30.0:
                break
            p_perturbed = p_start + rng.normal(0, 0.1, size=n_params)
            try:
                res = opt.minimize(
                    smooth_objective_fourier,
                    p_perturbed,
                    args=(50.0,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if np.isfinite(res.fun):
                    h_cand = fourier_to_heights(res.x)
                    val_cand = ratio_from_h(h_cand)
                    update_best(h_cand, val_cand)
            except Exception:
                pass

    # Phase 3: Direct Spatial Polish using L-BFGS-B on Heights
    # This is the "refinement machinery" from the Current Program, which is very effective
    # for shaving off local peaks in the autoconvolution.
    if time.time() < deadline - 10.0:
        h_start = best_h.copy()
        # Normalize to maintain scale? No, L2 normalization handles scale.
        # But L-BFGS-B might drift. We can fix the sum or just let it optimize.
        # Since the objective is scale-invariant (ratio), the optimizer might find a trivial solution
        # like scaling to 0 or infinity. 
        # To prevent this, we can add a penalty for deviation from sum=600 (mean=1) or just rely on bounds.
        # Better: Normalize h_start to have sum=N, and add a soft constraint or just monitor.
        # Actually, L-BFGS-B on a scale-invariant function can be tricky.
        # Let's add a small regularization term to keep sum close to N.
        
        def spatial_objective(h):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e6
            # Regularization to prevent scaling drift
            reg = 1e-6 * (s - N) ** 2
            return ratio_from_h(h) + reg

        try:
            res = opt.minimize(
                spatial_objective,
                h_start,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                h_cand = res.x
                val_cand = ratio_from_h(h_cand)
                update_best(h_cand, val_cand)
        except Exception:
            pass

    # Phase 4: Final Quick Coordinate Descent (from Current Program)
    # If time remains, do a few steps of coordinate descent on the best_h
    if time.time() < deadline - 2.0:
        current_h = best_h.copy()
        current_val = best_val
        step_size = 0.05
        indices = rng.choice(N, size=100, replace=False)
        
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for idx in indices:
                if time.time() > deadline - 2.0:
                    break
                original = current_h[idx]
                # Try increasing
                current_h[idx] = max(0, original + step_size)
                val_plus = ratio_from_h(current_h)
                if val_plus < current_val:
                    current_val = val_plus
                    improved = True
                else:
                    # Try decreasing
                    current_h[idx] = max(0, original - step_size)
                    val_minus = ratio_from_h(current_h)
                    if val_minus < current_val:
                        current_val = val_minus
                        improved = True
                    else:
                        current_h[idx] = original
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
            update_best(current_h, current_val)

    return list(best_h)
