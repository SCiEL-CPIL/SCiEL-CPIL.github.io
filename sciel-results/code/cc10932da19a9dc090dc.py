"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autoconv_fft(sig):
    """Compute full autoconvolution using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)
    return ac[:2*N-1]


def _score_and_grad(sig, beta=100.0):
    """
    Compute smooth surrogate score and gradient.
    Uses log-sum-exp as smooth max for |autocorr|.
    Score = (1/beta) * log(sum(exp(beta*|ac|))) / (sum(sig))^2
    """
    ac = _autoconv_fft(sig)
    abs_ac = np.abs(ac)
    
    # Smooth max via log-sum-exp
    max_val = np.max(abs_ac)
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * (abs_ac - max_val))))
    smooth_max = log_sum_exp / beta
    
    integral = np.sum(sig)
    if abs(integral) < 1e-10:
        return 1e10, np.zeros(N)
    
    score = smooth_max / (integral * integral)
    
    # Gradient computation
    # d(smooth_max)/d(ac_k) = beta * exp(beta*|ac_k|) / sum(exp(beta*|ac_j|)) * sign(ac_k)
    # But we need d(ac)/d(sig) which is the convolution kernel
    
    # For autoconvolution: ac[k] = sum_i sig[i] * sig[i+k]
    # d(ac[k])/d(sig[j]) = sig[j+k] + sig[j-k] (with appropriate bounds)
    
    # Weight for each autocorrelation lag
    weights = beta * np.exp(beta * (abs_ac - max_val)) / np.sum(np.exp(beta * (abs_ac - max_val)))
    
    # d(score)/d(sig[j]) involves summing over all lags
    # This is expensive; use FFT-based approach
    # 
    # ac = ifft(|fft(sig)|^2) -> gradient w.r.t. sig
    # d(ac[k])/d(sig[j]) = (sig * R)[j+k] + (sig * R)[j-k] where R is reversal
    # 
    # More simply: ac = sig ⊛ sig (autoconvolution)
    # d(ac)/d(sig) is a convolution operator
    # 
    # For the smooth max: grad_smooth_max = sum_k weights[k] * d(ac[k])/d(sig)
    # 
    # d(ac[k])/d(sig[j]) = sig[j-k] (if 0<=j-k<N) + sig[j+k] (if 0<=j+k<N)
    # This can be computed as: convolve(weights with sig) + convolve(weights_rev with sig)
    
    # Compute gradient of smooth_max w.r.t. sig
    # Using FFT: if we define g[k] = weights[k] * sign(ac[k]), then
    # grad_j = sum_k g[k] * (sig[j-k] + sig[j+k])
    # This is a convolution of g with sig (circular, with proper handling)
    
    g = weights * np.sign(ac)
    # Pad to handle linear convolution
    n_fft_g = 1
    while n_fft_g < 2*N - 1:
        n_fft_g *= 2
    
    # g has length 2N-1, sig has length N
    # Linear convolution: result has length 3N-2, we need indices 0..N-1
    # Actually, let's think differently.
    
    # ac[k] = sum_{i=0}^{N-1-k} sig[i] * sig[i+k] for k>=0
    # ac[k] = ac[-k] for negative k (by symmetry of autoconv)
    
    # d(ac[k])/d(sig[j]) = sig[j-k] (if 0 <= j-k <= N-1-k) + sig[j+k] (if 0 <= j+k <= N-1-k)
    
    # For efficiency, use the fact that:
    # sum_k g[k] * sig[j-k] = (g_rev ⊛ sig)[j] where g_rev is reversed
    # sum_k g[k] * sig[j+k] = (g ⊛ sig)[j]
    
    # Let's just do it directly with FFT
    # Create full g vector (length 2N-1, indexed -N+1 to N-1)
    g_full = np.zeros(2*N-1)
    g_full[N-1:] = g[N-1:]  # positive lags
    g_full[:N-1] = g[N-1:0:-1]  # negative lags (symmetric)
    
    # conv1[j] = sum_k g_full[k] * sig[j-k] for valid j
    # conv2[j] = sum_k g_full[k] * sig[j+k] for valid j
    
    # Use FFT for linear convolution
    n_fft_c = 1
    while n_fft_c < 3*N - 2:
        n_fft_c *= 2
    
    G = np.fft.rfft(g_full, n_fft_c)
    S = np.fft.rfft(sig, n_fft_c)
    
    # conv1: g_full ⊛ sig, take first N elements (shifted appropriately)
    # (g_full ⊛ sig)[j] = sum_k g_full[k] * sig[j-k]
    # g_full is centered at N-1, so this gives us the correlation
    conv1 = np.fft.irfft(G * S, n_fft_c)
    
    # The convolution result: (g_full ⊛ sig)[n] = sum_k g_full[k] * sig[n-k]
    # We want for each j: sum_k g_full[k] * sig[j-k] = (g_full ⊛ sig)[j]
    # But g_full has length 2N-1, sig has length N
    # Result has length 3N-2
    # (g_full ⊛ sig)[j] for j=0..N-1
    
    grad_conv1 = conv1[:N]
    
    # conv2: sum_k g_full[k] * sig[j+k]
    # = sum_k g_full[k] * sig[(N-1) - (N-1-j-k)]
    # Let sig_rev[i] = sig[N-1-i]
    # sum_k g_full[k] * sig[j+k] = sum_k g_full[N-1-k'] * sig_rev[N-1-j-k'] where k'=N-1-k
    # This is (g_full_rev ⊛ sig_rev)[N-1-j]
    
    sig_rev = sig[::-1]
    g_full_rev = g_full[::-1]
    
    G_rev = np.fft.rfft(g_full_rev, n_fft_c)
    S_rev = np.fft.rfft(sig_rev, n_fft_c)
    conv2 = np.fft.irfft(G_rev * S_rev, n_fft_c)
    
    # (g_full_rev ⊛ sig_rev)[m] = sum_k g_full_rev[k] * sig_rev[m-k]
    # We want sum_k g_full[k] * sig[j+k] for j=0..N-1
    # Let m = N-1-j, then sig_rev[m-k] = sig[N-1-m+k] = sig[j+k]
    # g_full_rev[k] = g_full[N-1-k]
    # So sum_k g_full[N-1-k] * sig[j+k] = (g_full_rev ⊛ sig_rev)[N-1-j]
    
    grad_conv2 = conv2[N-1:N-1+N][::-1]
    
    grad_smooth_max = grad_conv1 + grad_conv2
    
    # d(score)/d(sig[j]) = d(smooth_max)/d(sig[j]) / (int^2) - 2*smooth_max*sig[j] / int^3
    inv_int_sq = 1.0 / (integral * integral)
    inv_int_cu = 1.0 / (integral * integral * integral)
    
    grad = grad_smooth_max * inv_int_sq - 2.0 * smooth_max * sig * inv_int_cu
    
    return score, grad


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Start from constant signal (theoretical optimum for unconstrained case)
    sig = np.ones(N)
    
    # Try different initializations and optimize
    best_sig = sig.copy()
    best_score = _score_and_grad(sig)[0]
    
    # Use L-BFGS-B to optimize the smooth surrogate
    # Start with moderate beta for smoothness, then increase
    
    for beta in [50.0, 100.0, 200.0, 500.0]:
        if time.time() > deadline:
            break
        
        def objective(x):
            s, _ = _score_and_grad(x, beta=beta)
            return s
        
        def gradient(x):
            _, g = _score_and_grad(x, beta=beta)
            return g
        
        try:
            result = minimize(
                objective, 
                sig, 
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_sig = result.x.copy()
                sig = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Final polish with higher beta
    if time.time() < deadline - 10:
        for beta in [1000.0]:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    lambda x: _score_and_grad(x, beta=beta)[0], 
                    best_sig, 
                    jac=lambda x: _score_and_grad(x, beta=beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-9, 'ftol': 1e-12}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_sig = result.x.copy()
            except Exception:
                pass
    
    return best_sig
# EVOLVE-BLOCK-END
