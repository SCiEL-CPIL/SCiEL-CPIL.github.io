"""
Combined program: PSD spectral shaping + Annealed L-BFGS + Peak-Targeting Polish.
Structural change: reparameterize x via PSD (x = ifft(|H|^2) or similar) to ensure
x is an autocorrelation-like sequence (PSD), which is often optimal for minimizing
max autoconv relative to integral squared. Then optimize the spectral coefficients.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import rfft, irfft, rfftfreq

START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = rfft(x, n=L_fft)
    C = irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _psd_to_x(spec):
    """
    Convert a non-negative spectral magnitude vector to a time-domain signal x.
    We use x = ifft(|spec|^2) to ensure x is real and symmetric (autocorrelation of itself).
    Actually, let's just use x = ifft(spec) where spec is the PSD.
    To ensure x is real, spec must be symmetric.
    We optimize over the non-negative frequencies of the PSD.
    Let S[k] be the PSD. x[n] = ifft(S).
    We parametrize S by its non-negative part and enforce symmetry.
    For real x, S[k] = S[N-k].
    We optimize over S[0], S[1], ..., S[N//2].
    """
    # spec is length N//2 + 1
    # Construct full spectrum
    full_spec = np.zeros(N, dtype=complex)
    full_spec[:len(spec)] = spec
    # Enforce Hermitian symmetry for real signal
    # S[k] = conj(S[N-k])
    # Since we want S to be real and non-negative (PSD), S[k] = S[N-k]
    for k in range(1, len(spec)):
        if k != N - k: # Handle Nyquist if N is even
            full_spec[N-k] = spec[k]
        else:
            full_spec[k] = spec[k]
    
    x = irfft(full_spec)
    return x

def _x_to_psd(x):
    """Extract the non-negative frequency part of the PSD of x."""
    X = rfft(x)
    # PSD is |X|^2
    psd = np.abs(X)**2
    return psd

def _smooth_objective_and_grad_psd(spec, beta=50.0):
    """
    Objective and gradient for PSD parametrization.
    spec: array of non-negative PSD values for frequencies 0..N//2.
    """
    # Convert spec to x
    # To make it differentiable, we need the gradient of x w.r.t spec.
    # x = ifft(S) where S is constructed from spec.
    # This is a linear mapping.
    
    # Let's compute the gradient via the chain rule.
    # Objective f(x) = max(|ac(x)|) / (sum x)^2
    # d f / d spec = (d f / d x) @ (d x / d spec)
    
    # 1. Compute x and its objective/grad
    x = _psd_to_x(spec)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(spec)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    L = len(ac)
    
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / L
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        if sum_exp == 0:
            smooth_max = max_val
            weights = np.zeros_like(abs_ac)
            weights[np.argmax(abs_ac)] = 1.0
        else:
            smooth_max = (max_val + np.log(sum_exp)) / beta
            weights = exp_shifted / sum_exp
    
    val = smooth_max / (s * s)
    
    c = weights * np.sign(ac)
    
    # Gradient of smooth_max w.r.t x
    L_fft = 1
    while L_fft < 2 * L:
        L_fft *= 2
        
    c_padded = np.zeros(L_fft)
    c_padded[:L] = c
    x_padded = np.zeros(L_fft)
    x_padded[:N] = x
    
    C_fft = rfft(c_padded, n=L_fft)
    X_fft = rfft(x_padded, n=L_fft)
    
    B = irfft(C_fft * X_fft, n=L_fft)[:N]
    A = irfft(C_fft * np.conj(X_fft), n=L_fft)[:N]
    
    grad_smooth_max = A + B
    
    grad_x = grad_smooth_max / (s * s) - 2.0 * smooth_max / (s * s * s)
    
    # Now map grad_x to grad_spec
    # x = ifft(S)
    # S is constructed from spec.
    # S[0] = spec[0]
    # S[k] = spec[k] for 1 <= k <= N//2 (with symmetry handling)
    # x[n] = (1/N) * sum_k S[k] * exp(2j pi n k / N)
    # d x[n] / d spec[j] = (1/N) * (exp(2j pi n j / N) + (j != 0 and j != N/2 ? exp(-2j pi n j / N) : 0))
    # Wait, if we change spec[j], it affects S[j] and S[N-j].
    # If j=0, only S[0].
    # If j=N/2 (and N even), only S[N/2].
    # Else, S[j] and S[N-j].
    
    # grad_spec[j] = sum_n grad_x[n] * d x[n] / d spec[j]
    
    # d x[n] / d spec[j] is the real part of the basis vector.
    # Let's compute it numerically or analytically.
    # Analytical:
    # x = ifft(S)
    # S = M @ spec (where M is the mapping matrix)
    # x = ifft(M @ spec) = ifft(M) @ spec (linearity)
    # Let B = ifft(M). Then x = B @ spec.
    # grad_spec = B^T @ grad_x
    
    # Construct B
    # B[n, j] = d x[n] / d spec[j]
    
    # It's easier to just compute the gradient of x w.r.t spec using FFT properties.
    # x = irfft(S_full)
    # S_full is linear in spec.
    
    # Let's just compute B explicitly for N=400. It's 400x201.
    # B[n, 0] = 1/N
    # B[n, j] = (2/N) * cos(2 pi n j / N) for j > 0 and j < N/2
    # B[n, N/2] = 1/N * (-1)^n if N is even
    
    B_mat = np.zeros((N, len(spec)))
    B_mat[:, 0] = 1.0 / N
    
    for j in range(1, len(spec)):
        if N % 2 == 0 and j == N // 2:
            B_mat[:, j] = (1.0 / N) * ((-1) ** np.arange(N))
        else:
            B_mat[:, j] = (2.0 / N) * np.cos(2 * np.pi * np.arange(N) * j / N)
    
    grad_spec = B_mat.T @ grad_x
    
    return val, grad_spec

def _generate_initializations_psd():
    """Generate diverse PSD initializations."""
    inits = []
    t = np.linspace(0, 1, N)
    freqs = rfftfreq(N, d=1.0/N)
    
    # Flat PSD (white noise)
    inits.append(np.ones(N//2 + 1))
    
    # Low pass
    for cutoff in [0.1, 0.2, 0.5, 1.0]:
        psd = np.exp(-freqs**2 / cutoff**2)
        inits.append(psd)
        
    # High pass
    for cutoff in [0.1, 0.2, 0.5]:
        psd = freqs**2 / (freqs**2 + cutoff**2)
        inits.append(psd)
        
    # Band pass
    for center in [0.2, 0.5]:
        for width in [0.1, 0.2]:
            psd = np.exp(-(freqs - center)**2 / width**2)
            inits.append(psd)
            
    # Polynomial
    inits.append(1.0 + freqs**2)
    inits.append(1.0 / (1.0 + freqs**2))
    
    # Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        inits.append(rng.exponential(1.0, N//2 + 1))
        
    return inits

def construct():
    best_x = None
    best_val = np.inf
    
    # Phase 1: Evaluate initializations in PSD space
    inits = _generate_initializations_psd()
    for spec0 in inits:
        if _check_deadline():
            break
        x0 = _psd_to_x(spec0)
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
            best_spec = spec0.copy()
            
    if best_x is None:
        return np.ones(N)
        
    if _check_deadline():
        return best_x

    # Phase 2: Annealed L-BFGS Optimization in PSD space
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if _check_deadline():
            break
            
        def _obj_grad(spec, b=beta):
            return _smooth_objective_and_grad_psd(spec, beta=b)
        
        try:
            # Use current best spec as start point
            res = minimize(_obj_grad, best_spec, method='L-BFGS-B', jac=True,
                           options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-8})
            
            if res.fun < best_val:
                # Verify with exact objective
                x_trial = _psd_to_x(res.x)
                exact_val = _objective(x_trial)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = x_trial.copy()
                    best_spec = res.x.copy()
        except Exception:
            pass

    if _check_deadline():
        return best_x

    # Phase 3: Peak-Targeting Polish
    for _ in range(3):
        if _check_deadline():
            break
        def _obj_grad_high_beta(spec):
            return _smooth_objective_and_grad_psd(spec, beta=1000.0)
            
        try:
            res = minimize(_obj_grad_high_beta, best_spec, method='L-BFGS-B', jac=True,
                           options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-9})
            if res.fun < best_val:
                x_trial = _psd_to_x(res.x)
                exact_val = _objective(x_trial)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = x_trial.copy()
                    best_spec = res.x.copy()
        except Exception:
            pass
            
    # Phase 4: Random Perturbation Polish in PSD space
    rng = np.random.default_rng(42)
    step_size = 0.01
    for _ in range(1000):
        if _check_deadline():
            break
        perturbation = rng.normal(0, step_size, size=len(best_spec))
        spec_trial = best_spec + perturbation
        # Ensure non-negativity? L-BFGS-B handles bounds, but here we just perturb.
        # If we want to keep PSD valid, we should clip or use log-param.
        # For now, just perturb and let ifft handle it. If spec goes negative, x might be weird.
        # Better to clip spec to be non-negative.
        spec_trial = np.maximum(spec_trial, 0)
        
        x_trial = _psd_to_x(spec_trial)
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            best_spec = spec_trial.copy()
        else:
            if _ % 100 == 0:
                step_size *= 0.9

    return best_x
# EVOLVE-BLOCK-END
