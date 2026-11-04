import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 12000
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
    
    l2_coarse = np.sum(h * h)
    if l2_coarse < 1e-15:
        return float('inf')
        
    f = h[CELL]
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
        
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    max_conv = np.max(conv)
    
    return float(max_conv / l2)

def quick_ratio(h):
    """Optimized ratio for inner loops."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initial guess: constant function
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # === Phase 1: Sparse Analytic Signal Representation ===
    # Represent h as |sum c_k * exp(i * omega_k * x)|^2
    # This ensures non-negativity by construction.
    # We optimize the complex coefficients c_k and frequencies omega_k.
    
    # Number of terms in the sparse signal
    K = 6
    
    def signal_to_h(params):
        """
        params: [c1_re, c1_im, omega1, c2_re, c2_im, omega2, ..., cK_re, cK_im, omegaK]
        Returns h array of length N.
        """
        # Unpack parameters
        # Structure: for k in range(K): [re, im, omega]
        # Total params: 3*K
        
        # Reshape into (K, 3)
        P = params.reshape((K, 3))
        c_re = P[:, 0]
        c_im = P[:, 1]
        c = c_re + 1j * c_im
        omegas = P[:, 2]
        
        # Compute signal on X
        # x is shape (N,)
        # omegas is shape (K,)
        # We need sum_k c_k * exp(i * omega_k * x)
        
        # Matrix multiplication approach for speed
        # exp(i * omega_k * x_n) = cos(omega_k * x_n) + i * sin(omega_k * x_n)
        
        # Build matrix E where E[n, k] = exp(i * omega_k * x_n)
        # This is K x N or N x K. Let's do N x K.
        # angles = x[:, None] * omegas[None, :]  # (N, K)
        # E = np.exp(1j * angles)
        
        # For K=6 and N=600, this is small.
        angles = X[:, None] * omegas[None, :]
        E = np.exp(1j * angles)  # (N, K)
        
        # Signal s = E @ c
        s = E @ c  # (N,)
        
        # h = |s|^2
        h = np.abs(s) ** 2
        
        # Normalize? The ratio is scale-invariant, so we don't need to normalize.
        # But for numerical stability in optimization, maybe keep values in a reasonable range.
        # The optimizer will handle the scale.
        
        return h

    def sparse_ratio(params):
        h = signal_to_h(params)
        # Ensure non-negative (it is by construction, but clip just in case of numerical noise)
        h = np.clip(h, 0.0, None)
        return quick_ratio(h)

    # Multi-start optimization in sparse signal space
    n_sparse_starts = 20
    
    for start in range(n_sparse_starts):
        if time.time() > deadline - 60.0:
            break
        
        # Initialize parameters
        # c_k: random complex numbers
        # omega_k: random frequencies. 
        # The domain is [-0.25, 0.25]. 
        # Frequencies should be chosen to create interesting structures.
        # Let's pick frequencies from a set of harmonics or random.
        
        params = np.zeros(3 * K)
        
        for k in range(K):
            # Random complex coefficient
            params[3*k] = rng.uniform(-1.0, 1.0)
            params[3*k+1] = rng.uniform(-1.0, 1.0)
            
            # Random frequency. 
            # To get features on scale ~ 1/12, frequency ~ 12.
            # Let's sample frequencies from a range.
            # If we use too high frequencies, the step function approximation might be poor.
            # But the fine grid handles the convolution. The step function h is defined on N=600 bins.
            # High frequencies in x will be aliased in h if omega * dx > pi.
            # dx = 0.5/600 = 1/1200.
            # Nyquist freq = pi / dx = 600 * pi ~ 1884.
            # Let's try frequencies up to ~ 50.
            params[3*k+2] = rng.uniform(1.0, 50.0)
            
        # Ensure the sum of coefficients is not zero (trivial solution)
        # If all c_k are 0, h is 0. 
        # Let's add a small DC component or ensure one coefficient is non-zero.
        # Actually, if h is 0, ratio is inf. The optimizer will move away.
        
        try:
            # Use Powell for robustness
            result = opt.minimize(
                sparse_ratio,
                params,
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 500}
            )
            
            h_opt = signal_to_h(result.x)
            val_opt = result.fun
            update_best(h_opt, val_opt)
            
            # Refine with Nelder-Mead
            if time.time() < deadline - 50.0:
                result2 = opt.minimize(
                    sparse_ratio,
                    result.x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result2.fun < val_opt:
                    h_opt2 = signal_to_h(result2.x)
                    update_best(h_opt2, result2.fun)
                    
        except Exception:
            pass

    # === Phase 2: Spatial Domain Refinement ===
    # Refine the best solution found in the spatial domain.
    
    if time.time() < deadline - 10.0:
        cur_h = best_h.copy()
        cur_val = best_val
        
        step_size = 0.1
        min_step = 0.001
        stagnation_count = 0
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            center = int(rng.integers(0, N))
            window = 5
            start_idx = max(0, center - window)
            end_idx = min(N, center + window + 1)
            
            h_try = cur_h.copy()
            
            noise = rng.normal(0, step_size * np.max(cur_h), size=(end_idx - start_idx))
            h_try[start_idx:end_idx] = np.clip(cur_h[start_idx:end_idx] + noise, 0, None)
            
            v = quick_ratio(h_try)
            
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count > 100:
                    step_size *= 0.9
                    if step_size < min_step:
                        break
                    stagnation_count = 0

    # === Phase 3: Final Polishing ===
    if time.time() < deadline - 2.0:
        cur_h = best_h.copy()
        cur_val = best_val
        
        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
            
            noise = rng.normal(0, 0.01 * np.max(cur_h), size=N)
            h_try = np.clip(cur_h + noise, 0, None)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)

    return list(best_h)
