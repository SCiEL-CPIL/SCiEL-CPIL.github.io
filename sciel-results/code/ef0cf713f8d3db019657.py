"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct


def _autoconv_fft(h):
    """Compute full autoconvolution using FFT."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2*n-1]


def _objective_hard(h):
    """Hard objective for final evaluation."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _soft_max_obj(h, beta=50.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2 using soft-max."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        soft_max = m + np.log(np.sum(np.exp(shifted))) / beta
        return 2.0 * N * soft_max / (s * s)
    else:
        return 2.0 * N * m / (s * s)


def _soft_max_grad(h, beta=50.0):
    """Gradient of soft-max objective."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _autoconv_fft(h)
    m = np.max(conv)
    
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        soft_max = m + np.log(sum_exp) / beta
        grad_conv = exp_shifted / sum_exp
        
        # Gradient via FFT-based cross-correlation
        n_conv = len(conv)
        n_fft2 = 1
        while n_fft2 < n_conv + N - 1:
            n_fft2 *= 2
        GC = np.fft.rfft(grad_conv, n=n_fft2)
        HH = np.fft.rfft(h, n=n_fft2)
        corr = np.fft.irfft(GC * np.conj(HH), n=n_fft2)[:n_conv]
        # grad_h[j] = 2 * sum_{k=0}^{N-1} grad_conv[j+k] * h[k]
        grad_h = 2.0 * corr[:N]
        
        val = 2.0 * N * soft_max / (s * s)
        grad_val = 2.0 * N / (s * s) * grad_h - 4.0 * N * soft_max / (s * s * s)
        return grad_val
    else:
        idx_max = np.argmax(conv)
        grad_h = np.zeros(N)
        for j in range(N):
            d = idx_max - j
            if 0 <= d < N:
                grad_h[j] = 2.0 * h[d]
        val = 2.0 * N * m / (s * s)
        grad_val = 2.0 * N / (s * s) * grad_h - 4.0 * N * m / (s * s * s)
        return grad_val


def _dct_optimize(h0, max_iter=200, beta=50.0):
    """Optimize using DCT parameterization."""
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _obj_and_grad(coeffs):
        h = _dct_to_h(coeffs)
        obj = _soft_max_obj(h, beta)
        grad_h = _soft_max_grad(h, beta)
        # Chain rule: d(coeffs)/d(h) via DCT
        grad_coeffs = dct(grad_h, type=2, norm='ortho')
        return obj, grad_coeffs
    
    try:
        res = minimize(_obj_and_grad, dct_coeffs.copy(),
                      method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.fun < _objective_hard(h0):
            return _dct_to_h(res.x)
    except Exception:
        pass
    return h0


def _random_walk_init(rng, N):
    """Generate random walk with negative drift as initialization."""
    drift = -0.01
    noise_scale = 0.5
    h = np.zeros(N)
    for i in range(N):
        if i > 0:
            h[i] = h[i-1] + drift + noise_scale * rng.standard_normal()
        else:
            h[i] = noise_scale * rng.standard_normal()
    return h


def _toeplitz_eig_init(rng, N, K=30):
    """Initialize using top K eigenvectors of random symmetric Toeplitz matrix."""
    # Generate random symmetric Toeplitz matrix
    first_row = rng.standard_normal(N)
    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            T[i, j] = first_row[abs(i - j)]
    T = (T + T.T) / 2.0
    
    # Get top K eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(T)
        # Take eigenvectors corresponding to largest eigenvalues
        idx = np.argsort(eigvals)[::-1][:K]
        V = eigvecs[:, idx]
        
        # Random coefficients
        coeffs = rng.standard_normal(K)
        h = V @ coeffs
        # Add baseline to ensure sum is not zero
        h = 1.0 + h
        return h
    except Exception:
        return rng.standard_normal(N) + 1.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal
    for _ in range(5):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 2. Comb patterns
    for period in [2, 3, 4, 5]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Higher frequency waves
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 4. Polynomial bases
    for deg in [1, 2, 3]:
        x = np.linspace(-1, 1, N)
        h0 = np.zeros(N)
        for d in range(deg + 1):
            h0 += rng.standard_normal() * x**d
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 5. Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # 6. DCT-shaped initializations
    for num_modes in [5, 10, 20]:
        h0 = np.zeros(N)
        coeffs = np.zeros(N)
        for k in range(num_modes):
            coeffs[k] = rng.standard_normal() * (1.0 / (k + 1))
        zm = idct(coeffs, type=3, norm='ortho')
        h0 = 1.0 + zm
        initializations.append(h0)
    
    # 7. Random walk initializations
    for _ in range(5):
        h0 = _random_walk_init(rng, N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 8. Toeplitz eigenvector initializations
    for _ in range(5):
        h0 = _toeplitz_eig_init(rng, N, K=30)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # Optimize each initialization
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_curr = h0.copy()
        
        # Try DCT optimization with analytic gradient
        if time.time() < deadline - 15.0:
            h_dct = _dct_optimize(h_curr, max_iter=50, beta=50.0)
            val_dct = _objective_hard(h_dct)
            if val_dct < _objective_hard(h_curr):
                h_curr = h_dct
        
        # Update best
        val_curr = _objective_hard(h_curr)
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 10.0:
        for beta in [100.0, 200.0, 500.0]:
            try:
                def _obj(h):
                    return _soft_max_obj(h, beta)
                def _grad(h):
                    return _soft_max_grad(h, beta)
                
                res = minimize(_obj, best_h, jac=_grad,
                              method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Final direct optimization with high beta
    if time.time() < deadline - 5.0:
        try:
            beta = 1000.0
            def _obj(h):
                return _soft_max_obj(h, beta)
            def _grad(h):
                return _soft_max_grad(h, beta)
            
            res = minimize(_obj, best_h, jac=_grad,
                          method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            val = _objective_hard(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
