import numpy as np
import time
from scipy.optimize import minimize

N = 600
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s

K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768

def _build_fine_grid(heights):
    """Build fine-sampled function from step heights."""
    return np.repeat(heights, K)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    score = max_conv / (integral_f ** 2)
    return score

def _smooth_objective_and_grad(heights, beta=50.0):
    """
    Smooth surrogate for the normalized max autoconvolution using log-sum-exp.
    Returns (value, gradient) with respect to heights.
    
    The autoconvolution at lag k is a quadratic form: c_k = sum_{i,j} h_i h_j * delta(i+j-k)
    We use log-sum-exp to smooth the max over lags.
    """
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return 1e10, np.zeros(N)
    
    h = np.maximum(heights, 0.0)
    n = len(h)
    w = 0.5 / n
    
    # Build fine grid and compute autoconvolution
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(h, K)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # The relevant lags are from 0 to M_STEP-1 (or so)
    # For a function supported on [0, M_STEP), autoconvolution is supported on [0, 2*M_STEP)
    # But we only need lags that correspond to valid shifts
    # The max of autoconvolution normalized by integral^2
    
    integral_f = np.sum(h) * w
    if integral_f <= 1e-10:
        return 1e10, np.zeros(N)
    
    # Normalize conv so that we're looking at conv / integral_f^2
    conv_norm = conv / (integral_f ** 2)
    
    # Use log-sum-exp for smooth max
    # We only care about lags where the convolution is meaningful
    # The autoconvolution of a function on [0, L] is on [0, 2L]
    # For our case, the meaningful lags are 0 to M_STEP-1
    max_idx = min(M_STEP, len(conv_norm))
    conv_relevant = conv_norm[:max_idx]
    
    # Log-sum-exp for smooth max
    max_val = np.max(conv_relevant)
    if max_val < 1e-15:
        return 1e10, np.zeros(N)
    
    # Soft max: sum(exp(beta * (x_i - max_val))) / beta + max_val
    shifted = beta * (conv_relevant - max_val)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + max_val
    
    # Gradient of log-sum-exp w.r.t. conv_relevant
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)  # softmax weights
    
    # Now compute gradient of conv_relevant w.r.t. h
    # conv[k] = sum_{i,j} h_i h_j * delta(i*K + j*K - k) approximately
    # Actually, let's use the FFT structure
    # conv = irfft(FFT(f)^2)
    # d(conv)/d(h) can be computed via backprop through FFT
    
    # Alternative: use the fact that conv is a quadratic form
    # Let's compute gradient numerically for the smooth part, or use analytic
    
    # For now, use a simpler approach: the gradient of the log-sum-exp of conv
    # d(log_sum_exp)/d(conv_k) = weights[k]
    # d(conv_k)/d(h_i) = 2 * sum_j h_j * kernel(i, j, k)
    
    # Let's use a more efficient approach: 
    # conv = f * f (autoconvolution)
    # d(conv)/d(h) involves the derivative of the quadratic form
    
    # Actually, let's just use the chain rule properly
    # conv = irfft(rfft(f)^2)
    # Let F = rfft(f), then conv = irfft(F^2)
    # d(conv)/d(f) = 2 * irfft(F * conj(dF/df)) ... this is getting complex
    
    # Simpler: use the fact that for autoconvolution, 
    # d(conv[k])/d(h[i]) = 2 * sum_j h[j] * delta(i*K + j*K - k) for the fine grid
    # But this is O(N^2) which is too slow
    
    # Let's use a different approach: compute gradient via FFT
    # The autoconvolution in frequency domain is |F|^2
    # We can compute the gradient using the adjoint
    
    # Actually, let's just use a finite difference approach for the gradient
    # of the smooth objective, which is O(N) FFT calls - too slow
    
    # Better: use the analytic gradient
    # conv = f * f
    # In frequency: C = F^2 (where F = FFT(f))
    # dC/dh = 2 * F * dF/dh
    # dF/dh is the FFT of the one-hot vector at position corresponding to h
    
    # Let's compute it properly
    # f_fine[i] = h[i//K] for i in [0, M_STEP)
    # So df_fine/dh[j] is a vector with K ones at positions j*K to (j+1)*K - 1
    
    # dF/dh[j] = FFT(df_fine/dh[j])
    # d(conv)/dh[j] = 2 * irfft(F * conj(dF/dh[j]))
    
    # This is O(N * M_FFT) which is 600 * 32768 ~ 20M, feasible
    
    # But wait, we need d(conv_norm)/dh, and conv_norm = conv / integral_f^2
    # integral_f = sum(h) * w
    # d(conv_norm)/dh = (d(conv)/dh * integral_f^2 - conv * 2 * integral_f * d(integral_f)/dh) / integral_f^4
    # = d(conv)/dh / integral_f^2 - 2 * conv * d(integral_f)/dh / integral_f^3
    # d(integral_f)/dh[j] = w
    
    # So d(conv_norm)/dh[j] = d(conv)/dh[j] / integral_f^2 - 2 * conv * w / integral_f^3
    
    # And the final gradient is sum_k weights[k] * d(conv_norm[k])/dh[j]
    
    # Let's compute d(conv)/dh[j] for all j
    # dF/dh[j] = FFT of a vector with K ones at positions j*K to (j+1)*K-1
    # This is a geometric series in frequency domain
    
    # Actually, let's just compute it directly for each j - it's O(N * M_FFT)
    # 600 * 32768 = ~20M operations, should be fine
    
    grad_h = np.zeros(N, dtype=np.float64)
    
    # Precompute: for each j, dF/dh[j] = FFT of rect(j*K, K)
    # We can compute this more efficiently
    
    # Actually, let's use a simpler approach: compute the gradient of the 
    # log-sum-exp objective using the chain rule with FFT
    
    # Let's define:
    # g = log_sum_exp(conv_norm)
    # dg/d(conv_norm[k]) = weights[k]
    # d(conv_norm[k])/d(h[j]) = d(conv[k])/d(h[j]) / integral_f^2 - 2*conv[k]*w/integral_f^3
    
    # So dg/d(h[j]) = sum_k weights[k] * [d(conv[k])/d(h[j]) / integral_f^2 - 2*conv[k]*w/integral_f^3]
    # = (1/integral_f^2) * sum_k weights[k] * d(conv[k])/d(h[j]) - (2*w/integral_f^3) * sum_k weights[k] * conv[k]
    
    # The second term is scalar * sum_k weights[k] * conv[k], which is the same for all j
    # Let S = sum_k weights[k] * conv[k]
    # Then dg/d(h[j]) = (1/integral_f^2) * sum_k weights[k] * d(conv[k])/d(h[j]) - (2*w*S)/integral_f^3
    
    # Now, sum_k weights[k] * d(conv[k])/d(h[j]) = sum_k weights[k] * 2 * sum_i h[i] * delta(i*K + j*K - k)
    # = 2 * sum_i h[i] * sum_k weights[k] * delta(i*K + j*K - k)
    # = 2 * sum_i h[i] * weights[i*K + j*K]  (if i*K + j*K < max_idx)
    
    # This is a convolution-like operation
    # Let W[k] = weights[k] for k in [0, max_idx)
    # Then sum_k weights[k] * d(conv[k])/d(h[j]) = 2 * sum_i h[i] * W[i*K + j*K]
    
    # This is like a cross-correlation of h with W sampled at multiples of K
    
    # Let's compute it
    W = np.zeros(M_FFT, dtype=np.float64)
    W[:max_idx] = weights[:max_idx]
    
    # For each j, we need sum_i h[i] * W[(i+j)*K]
    # Let's create an array V where V[m] = W[m*K] for m in [0, 2*N)
    # Then sum_i h[i] * W[(i+j)*K] = sum_i h[i] * V[i+j]
    # This is a correlation of h with V
    
    V = np.zeros(2 * N, dtype=np.float64)
    for m in range(2 * N):
        idx = m * K
        if idx < M_FFT:
            V[m] = W[idx]
    
    # Correlation: (h * V)[j] = sum_i h[i] * V[i+j]
    # Use FFT
    h_pad = np.zeros(4 * N, dtype=np.float64)
    h_pad[:N] = h
    V_pad = np.zeros(4 * N, dtype=np.float64)
    V_pad[:2*N] = V
    
    H = np.fft.fft(h_pad)
    Vf = np.fft.fft(V_pad)
    corr = np.fft.ifft(H * Vf).real[:2*N]
    
    # corr[j] = sum_i h[i] * V[i+j] for j in [0, 2*N)
    # We need this for j in [0, N)
    
    sum_w_dconv = 2.0 * corr[:N]
    
    S = np.dot(weights[:max_idx], conv_relevant)
    
    grad_h = sum_w_dconv / (integral_f ** 2) - (2.0 * w * S) / (integral_f ** 3)
    
    # Project to nonnegative orthant: only keep positive part
    # Actually, for L-BFGS-B with bounds, we should return the unconstrained gradient
    # The optimizer handles the bounds
    
    return log_sum_exp, grad_h

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform (known to give score 1.0)
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Strategy: Direct optimization of height array using smooth surrogate
    # and gradient-based method
    
    rng = np.random.default_rng(42)
    
    # Try multiple random restarts with different initial conditions
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        # Initial condition: uniform + small perturbation
        if restart == 0:
            h_init = np.ones(N, dtype=np.float64)
        else:
            # Create a smooth perturbation
            freq = rng.integers(1, 10)
            phase = rng.uniform(0, 2 * np.pi)
            perturbation = 0.1 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h_init = np.maximum(np.ones(N) + perturbation, 0.0)
        
        # Optimize using L-BFGS-B with smooth objective
        try:
            def objective_and_grad(x):
                val, grad = _smooth_objective_and_grad(x)
                return val, grad
            
            result = minimize(
                objective_and_grad,
                h_init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.0, 5.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
            )
            
            if result.fun < best_score:
                best_h = np.maximum(result.x, 0.0)
                best_score = compute_score(best_h)
        except Exception:
            pass
    
    # Final refinement: local search around best with smaller steps
    if time.time() - START_TIME < TIME_LIMIT - 5:
        rng2 = np.random.default_rng(123)
        step_size = 0.01
        for _ in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbation = rng2.normal(0, step_size, N)
            h_trial = best_h + perturbation
            h_trial = np.maximum(h_trial, 0.0)
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
            else:
                # Reduce step size if no improvement
                step_size *= 0.9
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
