import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 600
DEADLINE = 135.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _dct_basis(n, num_modes):
    """Compute DCT-II basis matrix for n points, num_modes modes."""
    k = np.arange(num_modes)[:, None]
    x = np.arange(n)[None, :]
    # DCT-II: cos(pi * k * (2*x + 1) / (2*n))
    basis = np.cos(np.pi * k * (2 * x + 1) / (2 * n))
    # Normalize each column to have unit L2 norm
    norms = np.sqrt(np.sum(basis ** 2, axis=1))
    norms[norms == 0] = 1.0
    basis = basis / norms
    return basis

def _dct_to_heights(coeffs, basis, n):
    """Convert DCT coefficients to height vector."""
    heights = basis @ coeffs
    return np.maximum(heights, 0.0)

def _smooth_max(conv, alpha):
    """Smooth approximation of max(conv) using log-sum-exp."""
    m = np.max(conv)
    if m < 0:
        m = 0
    exp_conv = np.exp(alpha * (conv - m))
    return m + np.log(np.sum(exp_conv)) / alpha

def _autoconv_and_grad(heights, M=None):
    """Compute autoconvolution and gradient of max autoconvolution w.r.t. heights."""
    if M is None:
        n = len(heights)
        M = 1
        while M < 2 * n - 1:
            M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:len(heights)] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    peak_idx = np.argmax(conv)
    
    # Gradient of conv[peak_idx] w.r.t. f[i] = 2*f[peak_idx - i] if 0 <= peak_idx - i < len(heights)
    grad = np.zeros(len(heights), dtype=np.float64)
    for i in range(len(heights)):
        j = peak_idx - i
        if 0 <= j < len(heights):
            grad[i] = 2.0 * heights[j]
    
    return conv, grad, peak_idx

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Compute M once for reuse
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    # Strategy: Use DCT basis representation to reduce dimensionality
    # h = B @ c where B is DCT basis, c are coefficients
    # Optimize over c (low-dimensional) instead of h (600-dim)
    
    # Precompute DCT basis
    NUM_MODES = 30  # Use first 30 DCT modes
    basis = _dct_basis(N, NUM_MODES)
    
    # Initial coefficients: project uniform onto basis
    # For uniform, only DC component (mode 0) is non-zero
    c_init = np.zeros(NUM_MODES)
    c_init[0] = np.sum(h) / np.sqrt(N)  # DC component
    
    def dct_objective_and_grad(c, alpha):
        """Compute objective and gradient in DCT coefficient space."""
        # Compute heights from coefficients
        heights_raw = basis @ c
        heights = np.maximum(heights_raw, 0.0)
        
        # Compute autoconvolution
        f = np.zeros(M, dtype=np.float64)
        f[:N] = heights
        
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        # Smooth max
        m = np.max(conv)
        if m < 0:
            m = 0
        exp_conv = np.exp(alpha * (conv - m))
        weights = exp_conv / np.sum(exp_conv)
        
        # Gradient of smooth_max w.r.t. heights
        W_full = np.fft.rfft(weights)
        F_full = np.fft.rfft(f)
        corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
        grad_h = 2.0 * corr[:N]
        
        total = np.sum(heights)
        w = 0.5 / N
        
        if total <= 1e-10:
            return 1e10, np.zeros(NUM_MODES)
        
        # Objective
        sm = m + np.log(np.sum(exp_conv)) / alpha
        obj = sm * N / (total * total * w)
        
        # Gradient w.r.t. heights
        obj_grad_h = grad_h * (N / (total * total * w)) - sm * (2.0 * N / (total * total * total * w))
        
        # Gradient w.r.t. coefficients: grad_c = B^T @ grad_h (for positive part)
        # Only include gradient where heights_raw > 0 (due to max(0, .))
        mask = heights_raw > 1e-10
        obj_grad_c = basis.T @ (obj_grad_h * mask)
        
        return obj, obj_grad_c
    
    # Phase 1: Optimize in DCT coefficient space
    alpha_schedule = [50.0, 100.0, 200.0, 400.0, 800.0]
    
    for alpha_val in alpha_schedule:
        if time_left() < 20:
            break
        
        c_opt = c_init.copy()
        
        def objective_wrapper(c):
            obj, _ = dct_objective_and_grad(c, alpha_val)
            return obj
        
        def gradient_wrapper(c):
            _, grad = dct_objective_and_grad(c, alpha_val)
            return grad
        
        try:
            result = minimize(
                objective_wrapper,
                c_opt,
                jac=gradient_wrapper,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.x is not None:
                h_new = _dct_to_heights(result.x, basis, N)
                score = compute_score(h_new)
                if score < best_score:
                    best_score = score
                    best_h = h_new.copy()
                    c_init = result.x.copy()
        except Exception:
            pass
        
        # Try random restarts in coefficient space
        for restart in range(5):
            if time_left() < 15:
                break
            
            rng = np.random.default_rng(42 + restart * 7)
            c_rand = np.zeros(NUM_MODES)
            c_rand[0] = np.sum(h) / np.sqrt(N) * rng.uniform(0.5, 1.5)
            # Add small random components
            for k in range(1, min(15, NUM_MODES)):
                c_rand[k] = rng.normal(0, 0.1 * c_rand[0])
            
            try:
                result_r = minimize(
                    objective_wrapper,
                    c_rand,
                    jac=gradient_wrapper,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result_r.x is not None:
                    h_new_r = _dct_to_heights(result_r.x, basis, N)
                    score_r = compute_score(h_new_r)
                    if score_r < best_score:
                        best_score = score_r
                        best_h = h_new_r.copy()
                        c_init = result_r.x.copy()
            except Exception:
                pass
    
    # Phase 2: Full-space polish from best DCT result
    if time_left() > 30:
        h_opt = best_h.copy()
        
        def smooth_obj_and_grad_full(heights, alpha):
            """Compute smooth objective and gradient for full-space L-BFGS-B."""
            f = np.zeros(M, dtype=np.float64)
            f[:N] = heights
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M)
            
            sm = _smooth_max(conv, alpha)
            total = np.sum(heights)
            w = 0.5 / N
            if total <= 1e-10:
                return 1e10, np.zeros(N)
            
            # Gradient of smooth_max via log-sum-exp
            m = np.max(conv)
            if m < 0:
                m = 0
            exp_conv = np.exp(alpha * (conv - m))
            weights = exp_conv / np.sum(exp_conv)
            
            # Use FFT for efficient correlation
            W_full = np.fft.rfft(weights)
            F_full = np.fft.rfft(f)
            corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
            grad_f = 2.0 * corr[:N]
            
            S = total
            obj = sm * N / (S * S * w)
            obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
            
            return obj, obj_grad
        
        alpha_full = 500.0
        
        def objective_full(h):
            obj, _ = smooth_obj_and_grad_full(h, alpha_full)
            return obj
        
        def gradient_full(h):
            _, grad = smooth_obj_and_grad_full(h, alpha_full)
            return grad
        
        try:
            result_f = minimize(
                objective_full,
                h_opt,
                jac=gradient_full,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result_f.x is not None and np.all(result_f.x >= -1e-10):
                h_new_f = np.maximum(result_f.x, 0.0)
                score_f = compute_score(h_new_f)
                if score_f < best_score:
                    best_score = score_f
                    best_h = h_new_f.copy()
        except Exception:
            pass
    
    # Phase 3: Try structured shapes to see if any beat DCT result
    if time_left() > 10:
        x = np.linspace(0, 1, N)
        
        structured_candidates = []
        
        # Tapered edges (cosine taper)
        for taper_frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
            h_taper = np.ones(N)
            taper_len = int(taper_frac * N)
            if taper_len > 1:
                ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
                h_taper[:taper_len] = ramp
                h_taper[-taper_len:] = ramp[::-1]
            structured_candidates.append(h_taper)
        
        # Parabolic
        for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h_par = 1.0 - a * (2 * x - 1) ** 2
            h_par = np.maximum(h_par, 0.0)
            if np.max(h_par) > 0:
                structured_candidates.append(h_par)
        
        # Linear taper
        for slope in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h_lin = 1.0 - slope * np.abs(2 * x - 1)
            h_lin = np.maximum(h_lin, 0.0)
            if np.max(h_lin) > 0:
                structured_candidates.append(h_lin)
        
        # Plateau with smooth edges
        for plateau_frac in [0.3, 0.5, 0.7, 0.8]:
            h_plat = np.ones(N)
            edge_len = int((1 - plateau_frac) * N / 2)
            if edge_len > 1:
                ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
                h_plat[:edge_len] = ramp
                h_plat[-edge_len:] = ramp[::-1]
            structured_candidates.append(h_plat)
        
        for h_cand in structured_candidates:
            if time_left() < 3:
                break
            h_cand = np.maximum(h_cand, 0.0)
            if np.sum(h_cand) <= 0:
                continue
            score = compute_score(h_cand)
            if score < best_score:
                best_score = score
                best_h = h_cand.copy()
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
