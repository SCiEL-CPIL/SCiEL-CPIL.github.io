"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
Uses a hybrid approach:
1. Global search in a low-dimensional Fourier coefficient space using Differential Evolution.
2. Local refinement in the full space using Log-Sum-Exp smoothed gradient descent (L-BFGS-B).
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import gaussian_filter1d

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT sizes for fast convolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

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
        fft_size_cc = 1
        while fft_size_cc < N + 2 * N - 1:
            fft_size_cc *= 2
            
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        
        # Extract the relevant part
        grad_lse = 2.0 * conv_result[N-1:2*N-1]
        
        return grad_lse

    def smooth_objective_and_grad(x, beta=80.0):
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
        f = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(f):
            return 1e10, np.zeros_like(h)
        
        # Gradient
        weights = exp_terms / sum_weights
        grad_lse = fast_cross_corr(weights, h)
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return f, grad

    def smooth_objective(x, beta=80.0):
        """Smooth surrogate using log-sum-exp of conv entries."""
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=80.0):
        """Analytic gradient of the smooth surrogate."""
        _, grad = smooth_objective_and_grad(x, beta)
        return grad

    # --- Phase 1: Global Search in Fourier Coefficient Space ---
    # We represent h as a sum of cosines: h[i] = sum_{k=1}^K a_k * cos(2*pi*k*i/N)
    # This ensures smoothness and band-limiting, which helps reduce auto-correlation peaks.
    
    t = np.arange(N) / N
    # Choose number of Fourier components to optimize
    K = 20 
    # Precompute the basis matrix
    # Basis shape: (N, K)
    basis = np.zeros((N, K))
    for k in range(1, K + 1):
        basis[:, k-1] = np.cos(2 * np.pi * k * t)
    
    def fourier_to_h(coeffs):
        """Convert Fourier coefficients to time-domain signal."""
        return basis @ coeffs

    def fourier_objective(coeffs):
        """Objective function for the Fourier coefficients."""
        h = fourier_to_h(coeffs)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val

    # Differential Evolution bounds for coefficients
    # Initial guess magnitudes around 1.0
    bounds = [(-5.0, 5.0) for _ in range(K)]
    
    # Run Differential Evolution
    # We limit the time spent here to allow for the refinement phase.
    fe_limit = 2000  # Function evaluations limit
    max_iter = 100
    
    try:
        if time.time() - start_time < DEADLINE - 60: # Leave 60s for refinement
            de_result = differential_evolution(
                fourier_objective,
                bounds,
                seed=42,
                maxiter=max_iter,
                polish=False, # We will do our own polish
                tol=1e-8,
                init='sobol',
                update_interval=10,
                callback=None
            )
            
            if de_result.success or de_result.fun < best_value:
                best_coeffs = de_result.x
                h_candidate = fourier_to_h(best_coeffs)
                val, valid = evaluate(h_candidate)
                if valid and val < best_value:
                    best_value = val
                    best_h = h_candidate.copy()
    except Exception as e:
        pass

    # --- Phase 2: Local Refinement in Full Space ---
    # We take the best h from Phase 1 (or seed) and refine it using L-BFGS-B with LSE smoothing.
    
    # We can also try refining a few other good candidates from the DE population if available,
    # but for simplicity and speed, we stick to the best one found and maybe the seed.
    
    candidates_to_refine = []
    if 'best_h' in locals() and best_value < 2.0:
        candidates_to_refine.append(best_h.copy())
    
    # Also add the seed just in case DE failed or found something worse
    candidates_to_refine.append(np.ones(N).astype(float))
    
    # Add a few random smooth perturbations of the best found
    rng = np.random.RandomState(123)
    for _ in range(3):
        noise = rng.normal(0, 0.1, N)
        noise = gaussian_filter1d(noise, sigma=5)
        h_perturbed = best_h + noise
        s = np.sum(h_perturbed)
        if abs(s) > 1e-10:
            candidates_to_refine.append(h_perturbed)

    for init_h in candidates_to_refine:
        if time.time() - start_time > DEADLINE - 10:
            break

        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1

        try:
            # Step 1: Moderate beta for convergence
            result = minimize(
                lambda x: smooth_objective(x, beta=50.0),
                init_h,
                jac=lambda x: smooth_grad(x, beta=50.0),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            
            val, valid = evaluate(result.x)
            if valid and val < best_value:
                best_value = val
                best_h = result.x.copy()
                
            # Step 2: High beta for precise peak targeting
            if time.time() - start_time < DEADLINE - 5:
                result2 = minimize(
                    lambda x: smooth_objective(x, beta=150.0),
                    best_h,
                    jac=lambda x: smooth_grad(x, beta=150.0),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
                )
                
                val2, valid2 = evaluate(result2.x)
                if valid2 and val2 < best_value:
                    best_value = val2
                    best_h = result2.x.copy()
                    
        except Exception:
            continue

    # --- Phase 3: Exact Gradient Polish ---
    if time.time() - start_time < DEADLINE - 2:
        def exact_obj_grad(h):
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10, np.zeros_like(h)
            
            conv = fast_convolve(h)
            max_val = np.max(conv)
            idx = np.argmax(conv)
            
            raw = 2.0 * N * max_val / (s * s)
            val = abs(raw)
            
            dh_conv = np.zeros(N)
            km = idx - np.arange(N)
            mask = (km >= 0) & (km < N)
            dh_conv[mask] = h[km[mask]] + h[mask]
            
            grad = 2.0 * N * (dh_conv * s - 2.0 * max_val) / (s ** 3)
            
            if raw < 0:
                grad = -grad
            
            return val, grad

        try:
            result = minimize(
                exact_obj_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            
            val, valid = evaluate(result.x)
            if valid and val < best_value:
                best_value = val
                best_h = result.x.copy()
        except Exception:
            pass

    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = fast_convolve(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
