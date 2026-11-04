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

    # === Phase 1: Polynomial in z representation ===
    # Represent h(x) = |P(z(x))|^2 where z = exp(i * theta)
    # theta maps [-0.25, 0.25] to [0, 2*pi)
    # P(z) = sum_{k=0}^{D} c_k * z^k
    # This ensures non-negativity and smooth structure.
    
    D = 12  # polynomial degree
    
    def theta_of_x(x):
        return 2.0 * np.pi * (x + 0.25)
    
    THETA = theta_of_x(X)  # precomputed for the N points
    
    def poly_params_to_h(params):
        # params: 2*(D+1) real numbers representing complex coefficients
        # c_k = params[2k] + 1j * params[2k+1]
        n_coeffs = D + 1
        c = params[:2*n_coeffs].reshape(n_coeffs, 2)
        c = c[:, 0] + 1j * c[:, 1]
        
        # Build z powers: z_n^k for n in range(N), k in range(D+1)
        # z_n = exp(i * theta_n)
        z = np.exp(1j * THETA)  # (N,)
        
        # Build Vandermonde-like matrix V where V[n, k] = z_n^k
        # V has shape (N, D+1)
        # V[n, k] = z_n^k
        powers = np.arange(D + 1)
        V = np.power(z[:, None], powers[None, :])  # (N, D+1)
        
        # s = V @ c
        s = V @ c  # (N,)
        
        h = np.abs(s) ** 2
        return h
    
    def poly_ratio(params):
        h = poly_params_to_h(params)
        h = np.clip(h, 0.0, None)
        return quick_ratio(h)
    
    # Multi-start with Differential Evolution for global exploration
    n_de_starts = 5
    for start in range(n_de_starts):
        if time.time() > deadline - 80.0:
            break
        
        # Random initial population
        n_coeffs = D + 1
        pop_size = 20
        
        # Initial random coefficients
        c_re_init = rng.normal(0, 1.0, size=(pop_size, n_coeffs))
        c_im_init = rng.normal(0, 1.0, size=(pop_size, n_coeffs))
        
        # Interleave into single array
        pop = np.zeros((pop_size, 2 * n_coeffs))
        pop[:, 0::2] = c_re_init
        pop[:, 1::2] = c_im_init
        
        try:
            # Use Differential Evolution
            result = opt.differential_evolution(
                poly_ratio,
                bounds=[(-5.0, 5.0)] * (2 * n_coeffs),
                seed=start,
                maxiter=50,
                tol=1e-5,
                popsize=20,
                mutation=(0.3, 0.8),
                recombination=0.8,
                polish=False  # We'll polish separately
            )
            
            h_opt = poly_params_to_h(result.x)
            val_opt = result.fun
            update_best(h_opt, val_opt)
            
            # Refine with Powell
            if time.time() < deadline - 60.0:
                result2 = opt.minimize(
                    poly_ratio,
                    result.x,
                    method='Powell',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 300}
                )
                if result2.fun < val_opt:
                    h_opt2 = poly_params_to_h(result2.x)
                    update_best(h_opt2, result2.fun)
                    
        except Exception:
            pass
    
    # === Phase 2: Direct spatial refinement with soft-max ===
    # Use a soft approximation to max for differentiability
    def soft_ratio(h, beta=50.0):
        """Use soft-max approximation for smoother optimization."""
        f = h[CELL]
        F = np.fft.rfft(f, N_FFT)
        conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
        l2 = np.sum(f * f) * DX
        if l2 < 1e-15:
            return float('inf')
        
        # Soft max: (1/beta) * log(sum(exp(beta * x)))
        conv_clipped = np.clip(conv, -10.0, 10.0)
        soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * conv_clipped)))
        return float(soft_max / l2)
    
    if time.time() < deadline - 30.0:
        cur_h = best_h.copy()
        cur_val = best_val
        
        # Gradient-based refinement using L-BFGS-B
        # We need to compute gradients. For simplicity, use finite differences
        # or use scipy's minimize with numerical gradients
        
        def objective(h_flat):
            h = np.asarray(h_flat, dtype=np.float64)
            h = np.clip(h, 0.0, None)
            return soft_ratio(h, beta=50.0)
        
        try:
            result = opt.minimize(
                objective,
                cur_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = np.clip(result.x, 0.0, None)
            val_opt = quick_ratio(h_opt)  # Use exact ratio for evaluation
            update_best(h_opt, val_opt)
            
        except Exception:
            pass
    
    # === Phase 3: Local spatial search ===
    if time.time() < deadline - 10.0:
        cur_h = best_h.copy()
        cur_val = best_val
        
        step_size = 0.1
        min_step = 0.001
        stagnation_count = 0
        
        max_iters = 3000
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            center = int(rng.integers(0, N))
            window = 5
            start_idx = max(0, center - window)
            end_idx = min(N, center + window + 1)
            
            h_try = cur_h.copy()
            
            noise = rng.normal(0, step_size * np.max(cur_h) if np.max(cur_h) > 0 else 0.1, size=(end_idx - start_idx))
            h_try[start_idx:end_idx] = np.clip(cur_h[start_idx:end_idx] + noise, 0, None)
            
            v = quick_ratio(h_try)
            
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count > 80:
                    step_size *= 0.9
                    if step_size < min_step:
                        break
                    stagnation_count = 0
    
    # === Phase 4: Final polishing ===
    if time.time() < deadline - 2.0:
        cur_h = best_h.copy()
        cur_val = best_val
        
        for _ in range(80):
            if time.time() > deadline - 1.0:
                break
            
            noise = rng.normal(0, 0.01 * (np.max(cur_h) if np.max(cur_h) > 0 else 1.0), size=N)
            h_try = np.clip(cur_h + noise, 0, None)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
    
    return list(best_h)
