import numpy as np
import time

N = 400
L = 2 * N - 1
L_fft = 1
while L_fft < L:
    L_fft *= 2

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_objective_and_grad(x, tau=0.1):
    """
    Compute a smooth approximation of max(|ac|)/(sum x)^2 and its gradient.
    Uses soft-max over |ac| values.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft-max approximation of max
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    # Weighted sum: w_i = exp((abs_ac_i - m)/tau) / sum(exp((abs_ac_j - m)/tau))
    shifted = (abs_ac - m) / tau
    # Clip to avoid overflow/underflow
    shifted = np.clip(shifted, -50, 0)
    weights = np.exp(shifted)
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return m / (s * s), np.zeros_like(x)
    weights /= w_sum
    
    # Soft max value
    soft_max = m + tau * np.log(w_sum)
    
    # Objective value
    val = soft_max / (s * s)
    
    # Gradient computation
    # d(soft_max)/dx = sum_i w_i * d|ac_i|/dx
    # |ac_i| gradient: sign(ac_i) * dac_i/dx
    # ac_i = sum_{j} x_j * x_{i-j} (with appropriate bounds)
    # dac_i/dx_k = x_{i-k} + x_{k-i} (for valid indices, 0 otherwise)
    
    # For efficiency, compute gradient via FFT
    # ac = ifft(fft(x)^2)
    # dac/dx_k: this is more complex. Let's use a different approach.
    # 
    # Alternative: use the fact that for autocorrelation,
    # d(ac_i)/dx_k = x_{i-k} + x_{k-i} where defined
    # 
    # We can compute the gradient of soft_max w.r.t. x as:
    # grad_soft_max = sum_i w_i * sign(ac_i) * (x convolved appropriately)
    # 
    # Actually, let's think more carefully.
    # ac_i = sum_j x_j x_{i-j}
    # d(ac_i)/dx_k = x_{i-k} + x_{k-i} (when both indices valid)
    # 
    # This is equivalent to: if we define y = sign(ac) * weights,
    # then grad = y convolved with x + x convolved with y (with appropriate handling)
    
    # Let's compute it as:
    # For each i, d(ac_i)/dx_k = x_{i-k} + x_{k-i}
    # So grad_k = sum_i w_i * sign(ac_i) * (x_{i-k} + x_{k-i})
    
    # Define sign_ac = sign(ac) * weights
    sign_ac = np.sign(ac) * weights
    
    # grad_k = sum_i sign_ac_i * x_{i-k} + sum_i sign_ac_i * x_{k-i}
    # = (sign_ac conv x)_k + (sign_ac conv x)_{-k} ... need to be careful with indexing
    
    # Actually: sum_i sign_ac_i * x_{i-k} is the cross-correlation of sign_ac and x evaluated at k
    # And sum_i sign_ac_i * x_{k-i} is also related
    
    # Let's use FFT to compute this efficiently
    # Define: g = sum_i sign_ac_i * x_{i-k} for k in range(-N+1, N)
    # This is the cross-correlation of sign_ac and x
    
    L_fft = 1
    L = 2 * N - 1
    while L_fft < L:
        L_fft *= 2
    
    # sign_ac is of length L (2N-1), x is of length N
    # We want: for each k in 0..N-1, sum_{i} sign_ac_i * x_{i-k} where x_j = 0 if j < 0 or j >= N
    
    # This is a convolution-like operation
    # Let's pad and use FFT
    sa_padded = np.zeros(L_fft)
    sa_padded[:L] = sign_ac
    
    x_padded = np.zeros(L_fft)
    x_padded[:N] = x
    
    # Cross-correlation: sum_i sa_i * x_{i-k}
    # This is ifft(fft(sa) * conj(fft(x)))
    SA = np.fft.rfft(sa_padded, n=L_fft)
    X = np.fft.rfft(x_padded, n=L_fft)
    
    # Cross-correlation c[k] = sum_i sa[i] * x[i-k]
    # In FFT terms: c = ifft(fft(sa) * conj(fft(x)))
    corr = np.fft.irfft(SA * np.conj(X), n=L_fft)
    
    # corr[k] for k=0..L-1 gives sum_i sa[i] * x[i-k] (with x padded to 0)
    # We need grad_k for k=0..N-1
    # grad_k = corr[k] + corr[-k mod L_fft] ... let me think again
    
    # Actually: sum_i sign_ac_i * x_{k-i} = sum_i sign_ac_i * x_{-(i-k)} 
    # If we define x_neg[j] = x[-j] (reversed), then this is a correlation
    
    # Let me just compute it directly for clarity, but efficiently
    # grad_k = sum_i sign_ac[i] * (x[i-k] if 0<=i-k<N else 0) + sum_i sign_ac[i] * (x[k-i] if 0<=k-i<N else 0)
    
    # First term: sum_i sign_ac[i] * x[i-k] for i such that 0 <= i-k < N
    # = sum_{j=0}^{N-1} sign_ac[j+k] * x[j] (where j = i-k, so i = j+k)
    # Second term: sum_i sign_ac[i] * x[k-i] for i such that 0 <= k-i < N
    # = sum_{j=0}^{N-1} sign_ac[k-j] * x[j] (where j = k-i, so i = k-j)
    
    # So grad_k = sum_{j=0}^{N-1} x[j] * (sign_ac[j+k] + sign_ac[k-j])
    
    # This is like a convolution of x with a modified sign_ac
    
    # Let's compute using FFT
    # Define h[k] = sign_ac[k] for k in -N+1..N-1 (mapped to appropriate indices)
    # Actually, let's just do it with a loop over the FFT approach
    
    # Simpler: use the fact that this is a convolution
    # grad = x * (sign_ac_shifted) where sign_ac_shifted is defined appropriately
    
    # Let me just compute it numerically with a helper
    # For N=400, a direct O(N^2) computation might be too slow
    # Let's use FFT properly
    
    # The gradient of ac w.r.t. x can be computed as:
    # If ac = ifft(fft(x)^2), then d(ac)/dx is not straightforward
    # 
    # Let's use a different approach: finite differences on the soft-max
    
    # Actually, for the soft-max, the gradient is:
    # d(soft_max)/dx = sum_i w_i * d(|ac_i|)/dx
    # d(|ac_i|)/dx_k = sign(ac_i) * d(ac_i)/dx_k = sign(ac_i) * (x_{i-k} + x_{k-i})
    
    # So d(soft_max)/dx_k = sum_i w_i * sign(ac_i) * (x_{i-k} + x_{k-i})
    
    # Let's define v[i] = w_i * sign(ac_i)
    # Then d(soft_max)/dx_k = sum_i v[i] * x_{i-k} + sum_i v[i] * x_{k-i}
    
    # First sum: (v * x)[k] in correlation sense
    # Second sum: (v * x)[k] in another correlation sense
    
    # Both are the same: sum_i v[i] * x[i-k] = sum_i v[i] * x[k-i] if v is symmetric
    # But v is not necessarily symmetric
    
    # Let's compute both:
    # A[k] = sum_i v[i] * x[i-k]
    # B[k] = sum_i v[i] * x[k-i]
    
    # B[k] = sum_i v[i] * x[k-i] = sum_j v[k-j] * x[j] (substituting j=k-i)
    #      = (v conv x)[k] where v is treated as a filter
    
    # A[k] = sum_i v[i] * x[i-k] = sum_j v[j+k] * x[j] (substituting j=i-k)
    
    # So grad_k = A[k] + B[k]
    
    # A[k] = sum_{j=0}^{N-1} v[j+k] * x[j] for j+k in [0, L-1]
    # B[k] = sum_{j=0}^{N-1} v[k-j] * x[j] for k-j in [0, L-1]
    
    # Let's compute A and B using FFT
    # For A: it's like a cross-correlation
    # For B: it's like a convolution
    
    # Let's just compute the full gradient using a loop over k with vectorized inner ops
    # For N=400, this is 400 * 400 = 160K ops, which is fine
    
    v = np.sign(ac) * weights  # v[i] = w_i * sign(ac_i)
    
    grad_sm = np.zeros(N)
    for k in range(N):
        # A[k] = sum_{j=0}^{N-1} v[j+k] * x[j]
        idx_a = j_idx = np.arange(N)
        ia = idx_a + k
        valid_a = (ia >= 0) & (ia < L)
        term_a = np.sum(v[ia[valid_a]] * x[idx_a[valid_a]])
        
        # B[k] = sum_{j=0}^{N-1} v[k-j] * x[j]
        ib = k - idx_a
        valid_b = (ib >= 0) & (ib < L)
        term_b = np.sum(v[ib[valid_b]] * x[idx_a[valid_b]])
        
        grad_sm[k] = term_a + term_b
    
    # Now gradient of objective = soft_max / (s^2)
    # d/dx_k [soft_max / s^2] = (grad_sm_k * s^2 - soft_max * 2 * s) / s^4
    # = grad_sm_k / s^2 - 2 * soft_max / s^3
    
    grad_obj = grad_sm / (s * s) - 2 * soft_max / (s * s * s)
    
    return val, grad_obj

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 140  # safety margin before 150s kill

    best_x = np.ones(N)
    best_val = _objective(best_x)  # baseline: 1/N = 0.0025

    rng = np.random.default_rng(42)

    # Phase 1: Try structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
    ]
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]:
        initializations.append(r ** np.arange(N))
    for r in [1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    initializations.append(1.0 + 0.5 * np.cos(np.pi * t))
    initializations.append(1.0 + np.cos(np.pi * t))
    initializations.append(1.0 + t**2)
    initializations.append(1.0 + 4.0 * (t - 0.5)**2)
    initializations.append(np.exp(2.0 * t))
    initializations.append(np.exp(-2.0 * t))
    initializations.append(1.0 + 0.3 * np.sin(2 * np.pi * t))
    initializations.append(1.0 + 0.5 * np.sin(np.pi * t))
    
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        initializations.append(np.sign(np.sin(2 * np.pi * freq * t)))
        initializations.append(1.0 + 0.5 * np.sign(np.sin(2 * np.pi * freq * t)))
    
    for x0 in initializations:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    # Phase 2: Gradient descent with smooth surrogate (FFT-based gradient)
    tau = 0.05
    lr = 0.1
    for gd_iter in range(50000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        val_s, grad = _smooth_objective_and_grad(best_x, tau)
        if not np.isfinite(val_s):
            break
        
        actual_val = _objective(best_x)
        
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-10:
            break
        
        step = grad / (norm_grad + 1e-10)
        
        # Line search
        alpha = lr
        improved = False
        for _ in range(15):
            x_trial = best_x - alpha * step
            trial_val = _objective(x_trial)
            if trial_val < actual_val:
                best_x = x_trial.copy()
                best_val = trial_val
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-10:
                break
        else:
            lr = min(1.0, lr * 1.05)
        
        # Reduce tau for sharper max approximation
        if gd_iter % 200 == 0 and tau > 0.005:
            tau *= 0.85

    # Phase 3: Fine-tuning with smaller perturbations
    step_size = 0.005
    stall_count = 0
    for iteration in range(200000):
        if time.time() - START_TIME > TIME_LIMIT:
            break

        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation

        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.0005, step_size * 0.999)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(0.1, step_size * 1.05)

    return best_x
