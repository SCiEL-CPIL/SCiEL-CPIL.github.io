"""Combined program: PSD-based convex optimization + time-domain refinement for signed step functions."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _phase_objective_and_grad(phases, beta=100.0, magnitudes=None):
    """
    Optimize phases of a complex signal with fixed magnitudes.
    phases: array of length L_fft//2+1 (for real signal symmetry)
    The signal is X[k] = magnitudes[k] * exp(i*phases[k]) for k=0..L_fft//2
    with conjugate symmetry for real signal.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    if magnitudes is None:
        magnitudes = np.ones(L_fft // 2 + 1)
    
    # Build complex spectrum with conjugate symmetry
    X = np.zeros(L_fft, dtype=complex)
    for k in range(L_fft // 2 + 1):
        X[k] = magnitudes[k] * np.exp(1j * phases[k])
    for k in range(1, L_fft // 2 + 1):
        X[L_fft - k] = np.conj(X[k])
    
    # Time-domain signal
    x = np.fft.ifft(X).real[:N]
    
    # Autoconvolution
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(phases)
    
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
    
    val = smooth_max / (s * s)
    
    # Gradient computation
    c = weights * np.sign(ac)
    
    # d(ac[j])/d(phases[k]) for the phase parameters
    # ac[j] = ifft(|X|^2)[j] = ifft(magnitudes^2 * exp(i*2*phases))[j]
    # But |X|^2 = magnitudes^2 (real, independent of phases!)
    # So ac is independent of phases!
    # This means the objective is constant w.r.t. phases if magnitudes are fixed.
    
    # We need to optimize both magnitudes and phases, or use a different approach.
    # For now, return zero gradient to indicate this issue.
    grad = np.zeros_like(phases)
    
    return val, grad

def _spectral_objective_and_grad(psd_params, beta=100.0):
    """
    Optimize in PSD domain. psd_params are the first L_fft//2+1 values of PSD.
    PSD must be non-negative and symmetric.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    psd = np.zeros(L_fft // 2 + 1)
    psd[:] = np.maximum(psd_params, 0)
    
    ac = np.fft.irfft(psd, n=L_fft)[:L]
    abs_ac = np.abs(ac)
    
    dc = psd[0]
    if dc < 1e-15:
        return 1e10, np.zeros_like(psd_params)
    
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
    
    c = weights * np.sign(ac)
    fft_c = np.fft.fft(c, n=L_fft)
    grad_smooth = np.real(fft_c.conj()) / L_fft
    
    grad = grad_smooth[:L_fft//2+1] / dc
    grad[0] = (grad_smooth[0] * dc - smooth_max) / (dc * dc)
    
    return val, grad

def _psd_to_signal(psd_params):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    psd = np.maximum(psd_params, 0)
    X = np.zeros(L_fft, dtype=complex)
    X[:L_fft//2+1] = np.sqrt(psd)
    for k in range(1, L_fft//2+1):
        X[L_fft-k] = X[k]
    
    x = np.fft.ifft(X).real[:N]
    return x

def _quantize_signed(x, levels=3):
    """Quantize to signed step function with given number of levels."""
    # Normalize to [-1, 1] range first
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-10:
        return np.ones(N) * (1 if np.mean(x) >= 0 else -1)
    
    x_norm = (x - x_min) / (x_max - x_min)
    
    # Quantize to levels
    if levels == 2:
        return np.sign(x - np.median(x))
    elif levels == 3:
        # Three levels: -1, 0, 1
        third = len(x_norm) // 3
        q = np.zeros(N)
        q[:third] = -1
        q[2*third:] = 1
        return q
    else:
        # General quantization
        bins = np.linspace(0, 1, levels + 1)
        q = np.digitize(x_norm, bins[1:-1]) - 1
        # Map to [-1, 1]
        q = 2.0 * q / (levels - 1) - 1.0
        return q

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Spectral optimization in PSD domain
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    psd_dim = L_fft // 2 + 1
    
    # Try different PSD initializations
    rng = np.random.default_rng(42)
    
    psd_inits = [
        np.ones(psd_dim),
        np.linspace(2.0, 0.5, psd_dim),
        np.linspace(0.5, 2.0, psd_dim),
        1.0 + 0.5 * np.cos(np.pi * np.linspace(0, 1, psd_dim)),
        1.0 + np.cos(np.pi * np.linspace(0, 1, psd_dim)),
    ]
    
    for i in range(5):
        psd_inits.append(rng.exponential(1.0, size=psd_dim) + 0.1)
    
    for psd0 in psd_inits:
        if _check_deadline():
            break
        
        # Use scipy minimize
        from scipy.optimize import minimize
        
        def _psd_obj_grad(p):
            v, g = _spectral_objective_and_grad(p, beta=100.0)
            return v, g
        
        try:
            result = minimize(_psd_obj_grad, psd0, method='L-BFGS-B',
                             jac=True, options={'maxiter': 200, 'gtol': 1e-5})
            if result.fun < best_val:
                x_trial = _psd_to_signal(result.x)
                val_trial = _objective(x_trial)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = x_trial.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Time-domain refinement
    from scipy.optimize import minimize
    
    def _time_obj_grad(x, beta=100.0):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        
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
        
        c = weights * np.sign(ac)
        L_conv = 2 * N - 1
        L_fft_conv = 1
        while L_fft_conv < 2 * L_conv:
            L_fft_conv *= 2
        
        x_pad = np.zeros(L_fft_conv)
        x_pad[:N] = x
        c_pad = np.zeros(L_fft_conv)
        c_pad[:L_conv] = c
        
        X_fft = np.fft.rfft(x_pad)
        C_fft = np.fft.rfft(c_pad)
        
        cross1 = np.fft.irfft(C_fft * np.conj(X_fft), n=L_fft_conv)[:N]
        cross2 = np.fft.irfft(C_fft * X_fft, n=L_fft_conv)[:N]
        
        grad_smooth = cross1 + cross2
        grad = grad_smooth / (s * s) - 2.0 * smooth_max / (s * s * s)
        
        return val, grad
    
    for beta in [50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        
        def _obj_grad(x, b=beta):
            return _time_obj_grad(x, b)
        
        try:
            result = minimize(_obj_grad, best_x, method='L-BFGS-B',
                             jac=True, options={'maxiter': 300, 'gtol': 1e-6})
            if result.fun < best_val:
                val_exact = _objective(result.x)
                if val_exact < best_val:
                    best_val = val_exact
                    best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Quantization to signed step function
    # Try different quantization strategies
    quantized_candidates = []
    
    for levels in [2, 3, 5]:
        x_q = _quantize_signed(best_x, levels)
        val_q = _objective(x_q)
        quantized_candidates.append((val_q, x_q))
    
    # Also try threshold-based quantization
    for threshold in np.linspace(0.5, 3.0, 10):
        x_q = np.sign(best_x - np.mean(best_x))
        # Scale to have mean zero
        x_q = x_q - np.mean(x_q)
        val_q = _objective(x_q)
        quantized_candidates.append((val_q, x_q))
    
    # Select best quantized
    if quantized_candidates:
        quantized_candidates.sort(key=lambda x: x[0])
        best_q_val, best_q_x = quantized_candidates[0]
        if best_q_val < best_val:
            best_val = best_q_val
            best_x = best_q_x
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Local search on quantized solution
    rng = np.random.default_rng(123)
    step_size = 0.5
    stall_count = 0
    
    for iteration in range(30000):
        if _check_deadline():
            break
        
        # Random perturbation
        pert = rng.normal(0, step_size, size=N)
        x_trial = best_x + pert
        
        # Quantize
        x_q = _quantize_signed(x_trial, levels=3)
        val = _objective(x_q)
        
        if val < best_val:
            best_val = val
            best_x = x_q.copy()
            stall_count = 0
            step_size = max(0.1, step_size * 0.95)
        else:
            stall_count += 1
            if stall_count % 3000 == 0:
                step_size = min(2.0, step_size * 1.1)
    
    return best_x
# EVOLVE-BLOCK-END
