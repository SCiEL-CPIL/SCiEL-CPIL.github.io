import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _metric_offdiag(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _spectral_flatness(x):
    """
    Objective: minimize sum of |F_k|^2 for k != 0, normalized by S^2.
    By Parseval, this equals the sum of c_k^2 for k != 0.
    Minimizing this spreads energy across all lags, suppressing max|c_k|.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    off_energy = np.sum(np.abs(fx[1:]) ** 2)
    return off_energy / (s * s)

def _spectral_grad(x):
    """
    Gradient of spectral flatness w.r.t. x.
    For real x, the gradient of sum_{k!=0} |F_k|^2 / S^2 can be computed
    via the autoconvolution. The gradient of sum c_k^2 is 4 * autoconv(x) (off-diag).
    Actually, let's use the chain rule through FFT.
    d/dx_j [sum_{k=1}^{K} |F_k|^2] where F_k = sum_j x_j exp(-i 2 pi k j / size)
    = 2 Re[sum_{k=1}^{K} conj(F_k) * conj(exp(-i 2 pi k j / size))]
    = 2 Re[sum_{k=1}^{K} conj(F_k) * exp(i 2 pi k j / size)]
    This is the IFFT of [0, conj(F_1), ..., conj(F_K)] at index j.
    Since F is real-symmetric (x is real), F_k = conj(F_{size-k}).
    Let's just compute numerically or via the relation to autoconvolution.
    
    Actually, sum_{k=1}^{K} |F_k|^2 = sum_{j} sum_{l} x_j x_l sum_{k=1}^K exp(-i 2 pi k (j-l)/size)
    The inner sum is -1 if j-l is a multiple of size (and non-zero), else it's a Dirichlet kernel part.
    For size >= 2N-1, the only multiple of size in range -(N-1)..(N-1) is 0.
    So sum_{k=1}^K |F_k|^2 = sum_{j,l} x_j x_l * (-1)_{j=l} ... wait.
    The sum over k=1..K of exp(-i 2 pi k m / size) is -1 if m=0, else something else.
    Actually, sum_{k=0}^{size-1} exp(-i 2 pi k m / size) = size * delta_{m,0}.
    So sum_{k=1}^{size-1} exp(-i 2 pi k m / size) = -delta_{m,0}.
    Thus sum_{k=1}^{K} |F_k|^2 = sum_{j,l} x_j x_l (-delta_{j-l, 0}) = -sum_j x_j^2.
    Wait, this is for the full DFT length. But our FFT is padded to `size`.
    The relation is: sum_{k=0}^{size-1} |F_k|^2 = size * sum_j x_j^2 (Parseval with normalization 1).
    numpy.fft.rfft returns unnormalized forward, 1/size inverse.
    So sum_{k=0}^{size-1} |F_k|^2 = size * sum_j x_j^2.
    Therefore sum_{k=1}^{K} |F_k|^2 = size * sum_j x_j^2 - |F_0|^2 = size * sum x_j^2 - S^2.
    
    So the objective is (size * sum x_j^2 - S^2) / S^2 = size * sum x_j^2 / S^2 - 1.
    Minimizing this is equivalent to minimizing sum x_j^2 / S^2.
    Since S is fixed (we can normalize), minimizing sum x_j^2 is the same as minimizing the L2 norm.
    Wait, this is interesting. Minimizing the off-diagonal spectral energy is equivalent to minimizing the L2 norm of x, given fixed sum.
    But we want to minimize MAX |c_k|, not the sum of squares.
    The sum of squares of c_k (off-diag) is an upper bound on max|c_k|^2 only if there's one peak, but actually sum c_k^2 >= max c_k^2.
    So minimizing sum c_k^2 is a good proxy, but it might not minimize the max directly.
    
    Let's try a different objective: minimize the soft-max of |c_k|.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.ones_like(x) * 1e10
    c = _autoconv_lin(x)
    off = np.abs(c[1:]) if len(c) > 1 else np.array([0.0])
    if len(off) == 0:
        return np.zeros_like(x)
    m = np.max(off)
    if m < 1e-14:
        return np.zeros_like(x)
    # Soft-max gradient
    weights = np.exp(8.0 * (off - m)) / np.sum(np.exp(8.0 * (off - m)))
    # Gradient of soft-max of |c_k| w.r.t. x
    # c_k = sum_i x_i x_{i+k}
    # d c_k / d x_j = x_{j+k} + x_{j-k} (with boundary conditions)
    # This is essentially 2 * x * (circular conv with something)
    # Let's compute the gradient of sum_k w_k |c_k| w.r.t. x
    # d/dx_j sum_k w_k |c_k| = sum_k w_k sign(c_k) (x_{j+k} + x_{j-k})
    # This is a linear operation on x, so it's a convolution with a filter.
    # Let g_j = sum_k w_k sign(c_k) (x_{j+k} + x_{j-k})
    # This is equivalent to x convolved with a filter h where h_k = w_k sign(c_k) for k>0 and h_{-k} = w_k sign(c_k).
    # Since x is length N, and we use linear convolution, we need to be careful.
    # Let's construct the filter.
    K = len(off)
    signs = np.sign(c[1:])
    w = weights
    # h is length 2K+1, h[K] = 0, h[K+k] = w_k * signs_k, h[K-k] = w_k * signs_k
    h = np.zeros(2 * K + 1)
    h[K + 1:K + 1 + K] = w * signs
    h[0:K][::-1] = w * signs
    # Now g = linear_conv(x, h)
    g = np.convolve(x, h, mode='full')[:N]
    return g / (s * s)

def _optimize(x0, deadline):
    """
    Optimize using L-BFGS-B with the soft-max objective.
    """
    def objective(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10
        c = _autoconv_lin(x)
        off = np.abs(c[1:]) if len(c) > 1 else np.array([0.0])
        m = np.max(off)
        if m < 1e-14:
            return 0.0
        soft = m + np.log(np.sum(np.exp(8.0 * (off - m)))) / 8.0
        return soft / (s * s)
    
    def grad(x):
        return _spectral_grad(x)
    
    try:
        res = scipy.optimize.minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        return res.x
    except Exception:
        return x0

def _sign_round(x):
    """
    Round to sign vector with sum close to 1.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        s = 1.0
    # Normalize so sum is 1
    x_norm = x / s
    # Round to signs
    x_sign = np.sign(x_norm)
    # Fix zeros
    x_sign[x_sign == 0] = 1
    # Adjust sum
    current_sum = np.sum(x_sign)
    if abs(current_sum) < 1e-10:
        return x_sign
    # We want sum to be 1, but x_sign has entries +/-1.
    # This is a knapsack-like problem. Let's just scale back.
    return x_sign

def _local_refine(x, deadline):
    """
    Local search: try flipping individual signs to improve the metric.
    """
    best_x = x.copy()
    best_val = _metric_std(best_x)
    n = len(best_x)
    s = np.sum(best_x)
    
    for trial in range(500):
        if time.time() > deadline:
            break
        idx = np.random.randint(0, n)
        x_trial = best_x.copy()
        x_trial[idx] *= -1  # Flip sign
        s_trial = np.sum(x_trial)
        if abs(s_trial) < 1e-10:
            continue
        # Don't rescale, keep the sign vector as is
        val = _metric_std(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial
    
    return best_x

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Multiple starting points
    starts = [
        np.ones(N),
        rng.standard_normal(N),
        np.sin(np.linspace(0, 2 * np.pi * 3, N)),  # Higher frequency
        np.cos(np.linspace(0, 2 * np.pi * 7, N)),
        rng.choice([-1.0, 1.0], size=N).astype(float),
        np.tile([1, -1, 0, 0], N // 4)[:N],  # Structured pattern
    ]
    
    for start in starts:
        if time.time() > deadline - 60:
            break
        s = np.sum(start)
        if abs(s) < 1e-10:
            start = start + 1.0
            s = np.sum(start)
        start = start / s
        
        # Continuous optimization
        x_opt = _optimize(start, deadline)
        s_opt = np.sum(x_opt)
        if abs(s_opt) < 1e-10:
            continue
        x_opt = x_opt / s_opt
        val = _metric_std(x_opt)
        if val < best_val:
            best_val = val
            best_x = x_opt
    
    # Local refinement on the best continuous solution
    if time.time() < deadline - 10:
        x_ref = _local_refine(best_x, deadline)
        val_ref = _metric_std(x_ref)
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref
    
    return best_x
