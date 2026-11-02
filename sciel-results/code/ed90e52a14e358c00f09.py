"""Combined program: PSD-based convex optimization + time-domain refinement for signed step functions."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_objective_and_grad(x, beta=50.0):
    """Smooth surrogate using log-sum-exp for max(|ac|) / (sum x)^2, with gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Log-sum-exp for smooth max
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / len(abs_ac)
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    
    val = smooth_max / (s * s)
    
    # Gradient of smooth_max w.r.t. x:
    c = weights * np.sign(ac)
    
    # Term B: x[j] * sum_{k=j}^{j+N-1} c[k]
    c_padded = np.zeros(2 * N - 1)
    c_padded[:len(c)] = c
    c_cumsum = np.cumsum(np.concatenate([[0], c_padded]))
    window_sums = np.zeros(N)
    for j in range(N):
        k_lo = j
        k_hi = min(j + N, 2 * N - 1)
        window_sums[j] = c_cumsum[k_hi] - c_cumsum[k_lo]
    term_B = x * window_sums
    
    # Term A via FFT cross-correlation
    L_conv = 2 * N - 1
    L_fft_conv = 1
    while L_fft_conv < 2 * L_conv:
        L_fft_conv *= 2
    
    x_padded = np.zeros(L_fft_conv)
    x_padded[:N] = x
    
    cr = c[::-1]
    cr_padded = np.zeros(L_fft_conv)
    cr_padded[:len(cr)] = cr
    
    X_fft = np.fft.rfft(x_padded, n=L_fft_conv)
    CR_fft = np.fft.rfft(cr_padded, n=L_fft_conv)
    conv_result = np.fft.irfft(X_fft * CR_fft, n=L_fft_conv)
    
    term_A = conv_result[2*N-2 : 2*N-2-N : -1]
    
    grad_smooth_max = term_A + term_B
    
    grad = grad_smooth_max / (s * s) - 2.0 * smooth_max / (s * s * s)
    
    return val, grad

def _psd_objective_and_grad(psd, beta=100.0):
    """
    Objective in PSD domain: minimize smooth_max(|ifft(psd)|) / (sum_ifft(psd))^2
    
    psd is the power spectral density (real, non-negative).
    The time-domain signal is reconstructed via ifft of sqrt(psd) with appropriate phase.
    For simplicity, we use zero phase (all-positive reconstruction) and optimize psd directly.
    
    Actually, let's work directly: given psd (which is |X|^2), the autoconvolution is ifft(psd).
    We want to minimize max(|ifft(psd)|) / (sum(ifft(psd)))^2 subject to psd >= 0.
    
    But sum(ifft(psd)) = ifft(psd)[0] for linear convolution... wait, no.
    
    Let me reconsider. If x is the time-domain signal, then:
    - Autoconvolution ac = x * x (linear)
    - FFT(ac) = |FFT(x)|^2 = PSD
    - sum(x) = FFT(x)[0] (DC component)
    
    So we want: minimize max(|ifft(PSD)|) / (sqrt(PSD[0]))^2 = max(|ifft(PSD)|) / PSD[0]
    
    subject to PSD >= 0, and PSD must be a valid PSD (positive definite).
    
    For a finite sequence, any non-negative array is a valid PSD if we allow complex phases.
    But for real signals, PSD must be symmetric: PSD[k] = PSD[N-k].
    
    Let's parametrize PSD as a symmetric non-negative array.
    """
    # psd is already symmetric and non-negative (enforced by parametrization)
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # ifft of psd to get autoconvolution
    ac = np.fft.irfft(psd, n=L_fft)[:L]
    abs_ac = np.abs(ac)
    
    # DC component squared = (sum x)^2
    dc = psd[0]
    if dc < 1e-15:
        return 1e10, np.zeros_like(psd)
    
    # Smooth max
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / len(abs_ac)
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    
    val = smooth_max / dc
    
    # Gradient of smooth_max w.r.t. psd:
    # ac = ifft(psd), so dac/dpsd[k] = ifft basis vector
    # d(smooth_max)/dpsd[k] = sum_j weights[j] * sign(ac[j]) * d(ac[j])/dpsd[k]
    
    # d(ac[j])/dpsd[k] = (1/L_fft) * exp(2*pi*i*j*k/L_fft) for real case
    # Using FFT: ifft(psd)[j] = (1/L_fft) * sum_k psd[k] * exp(2*pi*i*j*k/L_fft)
    
    # Let c = weights * sign(ac)
    c = weights * np.sign(ac)
    
    # grad_psd = ifft(c) (appropriately scaled)
    # Actually: smooth_max = sum_j c[j] * ac[j] (approximately, for small beta)
    # d(smooth_max)/dpsd[k] = sum_j c[j] * d(ac[j])/dpsd[k]
    #                        = sum_j c[j] * (1/L_fft) * exp(2*pi*i*j*k/L_fft)
    #                        = (1/L_fft) * exp(2*pi*i*? * k/L_fft) ... 
    
    # More carefully: ac = ifft(psd), so ac[j] = (1/L_fft) * sum_k psd[k] * exp(2*pi*i*j*k/L_fft)
    # d(ac[j])/dpsd[k] = (1/L_fft) * exp(2*pi*i*j*k/L_fft)
    
    # d(smooth_max)/dpsd[k] = sum_j c[j] * (1/L_fft) * exp(2*pi*i*j*k/L_fft)
    #                        = (1/L_fft) * sum_j c[j] * exp(2*pi*i*j*k/L_fft)
    #                        = (1/L_fft) * fft(c)[k]  (with appropriate normalization)
    
    # np.fft.fft(c)[k] = sum_j c[j] * exp(-2*pi*i*j*k/L_fft)
    # We need sum_j c[j] * exp(+2*pi*i*j*k/L_fft) = fft(conj(c))[k] for real c
    # Since c is real: = conj(fft(c))[k]
    
    fft_c = np.fft.fft(c, n=L_fft)
    grad_smooth_max_psd = np.real(fft_c.conj()) / L_fft
    
    # Only the first L elements matter (psd has L_fft/2+1 elements for real)
    # psd is symmetric: psd[k] = psd[L_fft-k]
    
    # Gradient of val = smooth_max / dc
    # d(val)/dpsd[k] = (grad_smooth_max_psd[k] * dc - smooth_max * d(dc)/dpsd[k]) / dc^2
    # d(dc)/dpsd[k] = 1 if k==0 else 0
    
    grad = grad_smooth_max_psd / dc
    grad[0] = (grad_smooth_max_psd[0] * dc - smooth_max) / (dc * dc)
    
    # Enforce symmetry in gradient
    # For real PSD, grad[k] should equal grad[L_fft-k]
    half = L_fft // 2
    for k in range(1, half + 1):
        idx = L_fft - k
        if idx < len(grad):
            grad[k] = (grad[k] + grad[idx]) / 2
            if idx < len(grad):
                grad[idx] = grad[k]
    
    return val, grad[:L_fft//2+1]

def _psd_to_signal(psd):
    """Reconstruct time-domain signal from PSD (zero-phase, real)."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # For real signal, X[k] = sqrt(PSD[k]) * exp(i*phi[k])
    # Zero phase: X[k] = sqrt(PSD[k]) for k=0..L_fft/2, symmetric
    X = np.zeros(L_fft, dtype=complex)
    X[:L_fft//2+1] = np.sqrt(np.maximum(psd, 0))
    for k in range(1, L_fft//2+1):
        X[L_fft-k] = X[k]
    
    x = np.fft.ifft(X).real[:N]
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Try structured initializations in time domain
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
    initializations.append(np.ones(N) * np.sign(np.sin(np.pi * t)))
    initializations.append(np.ones(N) * np.sign(np.cos(np.pi * t)))
    initializations.append(1.0 + 0.5 * np.sin(2 * np.pi * t))
    initializations.append(1.0 - 0.5 * np.sin(2 * np.pi * t))
    initializations.append(np.ones(N) * (1.0 + 0.3 * np.sin(4 * np.pi * t)))
    initializations.append(np.ones(N) * (1.0 + 0.3 * np.cos(4 * np.pi * t)))
    initializations.append(np.ones(N) * (1.0 - 0.3 * np.sin(2 * np.pi * t)))
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: PSD-based convex optimization
    # Initialize PSD from best time-domain signal
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    X_init = np.fft.rfft(best_x, n=L_fft)
    psd_init = np.abs(X_init)**2
    
    # Parametrize psd as exp(log_psd) to enforce positivity
    # log_psd has L_fft//2+1 free parameters (with symmetry)
    half_size = L_fft // 2 + 1
    
    def _log_psd_to_psd(log_psd):
        """Convert log-PSD to PSD with symmetry."""
        psd_full = np.zeros(L_fft//2+1)
        psd_full = np.exp(log_psd)
        return psd_full
    
    def _psd_obj_and_grad(log_psd, beta=100.0):
        """Objective and gradient in log-PSD parametrization."""
        psd = _log_psd_to_psd(log_psd)
        val, grad_psd = _psd_objective_and_grad(psd, beta=beta)
        
        # Chain rule: d(log_psd)/dpsd = 1/psd
        grad_log_psd = grad_psd / np.maximum(psd, 1e-15)
        
        return val, grad_log_psd
    
    # Optimize in log-PSD space
    log_psd_init = np.log(np.maximum(psd_init, 1e-15))
    
    for beta in [50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        
        try:
            result = minimize(_psd_obj_and_grad, log_psd_init, method='L-BFGS-B',
                             jac=True, args=(beta,),
                             options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-10})
            
            if result.fun < best_val:
                # Reconstruct time-domain signal
                psd_opt = _log_psd_to_psd(result.x)
                x_opt = _psd_to_signal(psd_opt)
                
                # Normalize to ensure sum is reasonable
                s = np.sum(x_opt)
                if abs(s) > 1e-10:
                    # The objective is scale-invariant in a specific way,
                    # but let's just evaluate
                    exact_val = _objective(x_opt)
                    if exact_val < best_val:
                        best_val = exact_val
                        best_x = x_opt.copy()
                        log_psd_init = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Time-domain refinement using smooth surrogate
    for beta in [50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        def _obj_grad_beta(x, b=beta):
            val, grad = _smooth_objective_and_grad(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_grad_beta, best_x, method='L-BFGS-B',
                             jac=True, options={'maxiter': 300, 'gtol': 1e-5, 'ftol': 1e-8})
            if result.fun < best_val:
                exact_val = _objective(result.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Random perturbation refinement
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    
    for iteration in range(30000):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.999)
        else:
            stall_count += 1
            if stall_count % 2000 == 0:
                step_size = min(0.5, step_size * 1.03)
                if stall_count % 10000 == 0:
                    big_pert = rng.normal(0, step_size * 5, size=N)
                    x_big = best_x + big_pert
                    val_big = _objective(x_big)
                    if val_big < best_val:
                        best_val = val_big
                        best_x = x_big.copy()
    
    return best_x
# EVOLVE-BLOCK-END
