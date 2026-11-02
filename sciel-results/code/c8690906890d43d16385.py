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
    # ac[j] = sum_{i} x[i] * x[j-i] for valid indices (linear autoconvolution)
    # d(ac[j])/d(x[k]) = x[j-k] + x[k-j] (with appropriate bounds)
    # d(smooth_max)/d(x[k]) = sum_j c[j] * (x[j-k] + x[k-j]) where c = weights * sign(ac)
    
    c = weights * np.sign(ac)
    L = 2 * N - 1
    
    # Compute gradient using FFT for efficiency
    # For each k, grad[k] = sum_j c[j] * (x[j-k] + x[k-j])
    # This is a cross-correlation: sum_j c[j] * x[j-k] + sum_j c[j] * x[k-j]
    
    # Use FFT to compute cross-correlations
    L_fft = 1
    while L_fft < 2 * L:
        L_fft *= 2
    
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x
    
    c_pad = np.zeros(L_fft)
    c_pad[:L] = c
    
    X_fft = np.fft.rfft(x_pad)
    C_fft = np.fft.rfft(c_pad)
    
    # cross1[k] = sum_j c[j] * x[j-k] = correlation of c and x
    # In FFT: ifft(conj(C) * X)[k] gives sum_j c[j] * x[j+k]
    # We want sum_j c[j] * x[j-k] = ifft(C * conj(X))[k]
    cross1 = np.fft.irfft(C_fft * np.conj(X_fft), n=L_fft)[:N]
    
    # cross2[k] = sum_j c[j] * x[k-j] = ifft(C * X)[k] (convolution-like)
    cross2 = np.fft.irfft(C_fft * X_fft, n=L_fft)[:N]
    
    grad_smooth_max = cross1 + cross2
    
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
    
    # Phase 2: Time-domain refinement using smooth surrogate with multiple betas
    # Use the corrected gradient computation
    for beta in [20.0, 50.0, 100.0, 200.0, 500.0]:
        if _check_deadline():
            break
        def _obj_grad_beta(x, b=beta):
            val, grad = _smooth_objective_and_grad(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_grad_beta, best_x, method='L-BFGS-B',
                             jac=True, options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-10})
            if result.fun < best_val:
                exact_val = _objective(result.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Multi-start refinement from perturbed versions of best solution
    rng = np.random.default_rng(42)
    for num_restarts in range(5):
        if _check_deadline():
            break
        
        # Create perturbed starting points
        for scale in [0.1, 0.3, 0.5, 1.0]:
            if _check_deadline():
                break
            x_pert = best_x + rng.normal(0, scale, size=N)
            val_pert = _objective(x_pert)
            if val_pert < best_val:
                best_val = val_pert
                best_x = x_pert.copy()
            
            # Refine from this starting point
            for beta in [50.0, 100.0, 200.0]:
                if _check_deadline():
                    break
                def _obj_grad_beta2(x, b=beta):
                    val, grad = _smooth_objective_and_grad(x, beta=b)
                    return val, grad
                
                try:
                    result = minimize(_obj_grad_beta2, best_x, method='L-BFGS-B',
                                     jac=True, options={'maxiter': 300, 'gtol': 1e-6, 'ftol': 1e-10})
                    if result.fun < best_val:
                        exact_val = _objective(result.x)
                        if exact_val < best_val:
                            best_val = exact_val
                            best_x = result.x.copy()
                except Exception:
                    pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Adaptive random perturbation refinement
    rng = np.random.default_rng(123)
    step_size = 0.1
    stall_count = 0
    max_iter = 50000
    
    for iteration in range(max_iter):
        if _check_deadline():
            break
        
        # Adaptive step size: decrease when improving, increase when stalling
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.95)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(1.0, step_size * 1.1)
                # Occasionally try a larger jump to escape local minima
                if stall_count % 20000 == 0:
                    big_pert = rng.normal(0, step_size * 3, size=N)
                    x_big = best_x + big_pert
                    val_big = _objective(x_big)
                    if val_big < best_val:
                        best_val = val_big
                        best_x = x_big.copy()
                        step_size = max(0.01, step_size * 0.5)
    
    return best_x
# EVOLVE-BLOCK-END
