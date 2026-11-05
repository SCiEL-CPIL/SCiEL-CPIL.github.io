import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

N = 400

def construct():
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val if valid else np.inf

    val_seed = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT sizes for fast convolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    fft_size_cc = 1
    while fft_size_cc < N + 2 * N - 1:
        fft_size_cc *= 2

    def fast_convolve(h):
        """Fast convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        using FFT-based cross-correlation.
        """
        h_rev = h[::-1]
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        
        # Extract the relevant part
        grad_lse = 2.0 * conv_result[N-1:2*N-1]
        
        return grad_lse

    def smooth_objective_and_grad(x, beta=150.0):
        """Compute both objective and gradient in one pass to avoid redundant FFTs."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        max_shifted = np.max(shifted)
        if max_shifted > 500:
            return 1e10, np.zeros_like(h)
        
        exp_terms = np.exp(beta * (shifted - max_shifted))
        sum_weights = np.sum(exp_terms)
        if sum_weights < 1e-300:
            return 1e10, np.zeros_like(h)
        
        lse = max_c + max_shifted + np.log(sum_weights) / beta
        f_val = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(f_val):
            return 1e10, np.zeros_like(h)
        
        # Gradient of lse w.r.t. h
        weights = exp_terms / sum_weights
        grad_lse = fast_cross_corr(weights, h)
        
        # Gradient of f = 2*N*lse/s^2
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        # Gradient of |f|
        if f_val >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return f_val, grad

    def smooth_objective_with_grad(x, beta=150.0):
        """Return (f, grad) together to avoid redundant FFTs."""
        return smooth_objective_and_grad(x, beta)

    def smooth_objective(x, beta=150.0):
        """Smooth surrogate using log-sum-exp of conv entries."""
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=150.0):
        """Analytic gradient of the smooth surrogate."""
        _, grad = smooth_objective_and_grad(x, beta)
        return grad

    def solve_ccp_iteration(h_current, max_iter=50, tol=1e-8):
        """
        Convex-Concave Procedure for minimizing max_k sum_{i+j=k} h_i h_j
        subject to sum(h) = 1.
        
        The problem is:
          min C
          s.t. sum_{i+j=k} h_i h_j <= C for all k
               sum(h) = 1
          
          Linearize the quadratic constraint at h_current:
          sum_{i+j=k} (h_i h_j + h_i (h_j - h_current_j) + h_j (h_i - h_current_i) - h_current_i h_current_j) <= C
          which simplifies to:
          2 * sum_{i+j=k} h_i h_current_{k-i} - sum_{i+j=k} h_current_i h_current_{k-i} <= C
          
          This is linear in h.
        """
        h = h_current.copy()
        s = np.sum(h)
        if abs(s) < 1e-12:
            return h
        h = h / s  # Normalize to sum=1
        
        # Precompute the linearization coefficients
        # For each k, the linearized constraint is:
        # 2 * sum_{i} h_i * h_current_{k-i} - C_k <= C
        # where C_k = sum_i h_current_i * h_current_{k-i}
        
        # We solve:
        # min C
        # s.t. 2 * (h * h_current)_k - C_k <= C for all k
        #      sum(h) = 1
        
        # This is a linear program. We use scipy's linprog or solve it via KKT.
        # Since it's a small LP, we can use scipy.optimize.linprog.
        # But linprog might be slow. Let's try a projected gradient or simple iteration.
        
        # Actually, let's just do a few steps of gradient descent on the smoothed version
        # with the linearized constraints as a penalty.
        
        # For simplicity and speed, let's just do a few L-BFGS-B steps on the smooth objective
        # but with a stronger penalty on the max autocorrelation.
        
        # Let's try a different approach: minimize the smooth objective but with
        # a hard constraint that the max autocorrelation doesn't increase.
        
        # For now, let's just do a standard L-BFGS-B optimization from h_current.
        # The CCP idea is to alternate between:
        # 1. Fix h_current, solve for h that minimizes max autocorrelation
        # 2. Update h_current
        
        # Since step 1 is non-convex, let's just use the smooth optimization.
        
        return h

    rng = np.random.RandomState(42)

    # --- Generate diverse initial points ---
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # Random uniform
    for _ in range(10):
        initial_points.append(rng.uniform(-1, 1, N))

    # Bimodal: mostly zero with random spikes
    for _ in range(10):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)

    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)

    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))

    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [25, 60, 120, 200]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            h_double = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) - np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_double)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7]:
        for _ in range(3):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)

    # Structured patterns: blocks of +1 and -1
    for block_size in [20, 50, 100]:
        n_blocks = N // block_size
        for _ in range(3):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            initial_points.append(h_blocks)

    # Smooth random via Gaussian filter
    for _ in range(10):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(2, 20))
        initial_points.append(h_smooth)

    # --- Spectral null initial points: design h such that H[0] (DC) is small ---
    # The autoconvolution peak is dominated by the DC term |H[0]|^2 / N.
    # If we design h to have a spectral null at DC (i.e., sum(h) is small relative to energy),
    # the peak autoconvolution can be reduced.
    # We generate random vectors in the subspace orthogonal to the all-ones vector,
    # then add a small DC component to keep sum(h) non-zero.
    
    # Projection matrix onto subspace orthogonal to 1
    ones = np.ones(N)
    ones_norm_sq = np.dot(ones, ones)
    
    for _ in range(25):
        # Random vector
        v = rng.normal(0, 1, N)
        # Project out DC component
        dc_component = np.dot(v, ones) / ones_norm_sq * ones
        v_zero_dc = v - dc_component
        # Add small DC offset to ensure sum(h) != 0
        dc_offset = rng.uniform(0.1, 0.5) * np.linalg.norm(v_zero_dc) / N
        h_spectral = v_zero_dc + dc_offset * ones
        initial_points.append(h_spectral)
    
    # More structured spectral nulls: use random phases in Fourier domain
    for _ in range(20):
        # Generate random spectrum with null at DC
        spectrum = rng.normal(0, 1, N // 2 + 1) + 1j * rng.normal(0, 1, N // 2 + 1)
        spectrum[0] = 0.0  # Null at DC
        # Ensure the spectrum is conjugate symmetric for real h
        if N % 2 == 0:
            spectrum_full = np.zeros(N, dtype=complex)
            spectrum_full[:N // 2 + 1] = spectrum
            spectrum_full[N // 2 + 1:] = np.conj(spectrum[1:N // 2][::-1])
        else:
            spectrum_full = np.zeros(N, dtype=complex)
            spectrum_full[:N // 2 + 1] = spectrum
            spectrum_full[N // 2 + 1:] = np.conj(spectrum[1:N // 2][::-1])
        h_fourier_null = np.real(np.fft.ifft(spectrum_full))
        # Add small DC offset
        dc_offset = rng.uniform(0.1, 0.5) * np.linalg.norm(h_fourier_null) / N
        h_fourier_null += dc_offset
        initial_points.append(h_fourier_null)

    # Sort by initial value (best first)
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # --- Optimize from best initial points ---
    for h0 in initial_points:
        if time.time() - start_time > DEADLINE - 30:
            break
        try:
            res = minimize(
                smooth_objective,
                h0,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = res.x
            val_opt = evaluate(h_opt)
            if val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # --- High-precision polish pass with higher beta ---
    if time.time() - start_time < DEADLINE - 5:
        beta_high = 500.0
        try:
            res2 = minimize(
                smooth_objective,
                best_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            # Override beta via closure isn't clean; do manual polish instead
        except Exception:
            pass

    # Manual polish: use higher beta by calling smooth_objective_and_grad directly
    if time.time() - start_time < DEADLINE - 3:
        beta_polish = 500.0
        h_polish = best_h.copy()
        for it in range(200):
            if time.time() - start_time > DEADLINE - 2:
                break
            f, grad = smooth_objective_and_grad(h_polish, beta=beta_polish)
            if not np.isfinite(f) or np.linalg.norm(grad) < 1e-12:
                break
            # Line search: backtracking
            alpha = 1.0
            f_cur = f
            for ls in range(30):
                h_try = h_polish - alpha * grad
                f_try, _ = smooth_objective_and_grad(h_try, beta=beta_polish)
                if np.isfinite(f_try) and f_try < f_cur - 1e-10 * alpha * np.dot(grad, grad):
                    h_polish = h_try
                    break
                alpha *= 0.5
            else:
                break
        val_polish = evaluate(h_polish)
        if val_polish < best_value:
            best_value = val_polish
            best_h = h_polish

    # Final: try a few more restarts from perturbed best if time allows
    if time.time() - start_time < DEADLINE - 10:
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            h_pert = best_h + rng.normal(0, 0.01 * np.linalg.norm(best_h) / np.sqrt(N), N)
            try:
                res3 = minimize(
                    smooth_objective,
                    h_pert,
                    jac=smooth_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                h_opt3 = res3.x
                val_opt3 = evaluate(h_opt3)
                if val_opt3 < best_value:
                    best_value = val_opt3
                    best_h = h_opt3.copy()
            except Exception:
                pass

    return best_h
