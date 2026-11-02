"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _autoconv_fft_fast(x):
    """Compute linear autoconvolution using minimal FFT size (2N-1 is enough)."""
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

def _objective_grad_smooth(x, beta=30.0):
    """
    Compute soft-max surrogate objective and its gradient.
    Uses log-sum-exp for numerical stability.
    Optimized FFT sizes for speed.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation using minimal FFT sizes
    g = weights * np.sign(ac)
    
    # For linear cross-correlation and convolution, we need FFT size >= 2N-1
    L_fft = 1
    while L_fft < 2 * N - 1:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[n-j] = cross-correlation
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n] = convolution
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _epigraph_objective(params, beta=50.0):
    """
    Epigraph formulation: minimize t subject to |ac[n]| <= t*s^2 for all n.
    We optimize over (x, t) jointly where t = max|ac| / s^2.
    
    params = [x_0, ..., x_{N-1}, t]
    """
    x = params[:N]
    t = params[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(params)
    
    s2 = s * s
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Constraint: |ac[n]| <= t * s^2, i.e., |ac[n]|/s^2 <= t
    # Soft penalty: sum exp(beta * (|ac[n]|/s^2 - t))
    violation = abs_ac / s2 - t
    shifted = beta * violation
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    penalty = np.sum(exp_shifted)
    
    # Objective: t + penalty/s^2 (normalized)
    val = t + penalty / s2
    
    # Gradient w.r.t. x:
    # d/dx_j [penalty/s^2] = (1/s^2) * sum_n exp(beta*(|ac[n]|/s^2 - t)) * sign(ac[n]) * d(ac[n])/dx_j
    # + (1/s^2) * sum_n exp(...) * (-2/s^2) * |ac[n]| * d(ac[n])/dx_j  [from derivative of |ac[n]|/s^2]
    # Wait, let me be more careful:
    # f = sum_n exp(beta * (|ac[n]|/s^2 - t))
    # d(f)/dx_j = sum_n exp(...) * beta * sign(ac[n]) * d(ac[n])/dx_j / s^2
    #             + sum_n exp(...) * beta * |ac[n]| * (-2/s^3) * 1 * d(ac[n])/dx_j  -- no
    # Actually: d(|ac[n]|/s^2)/dx_j = sign(ac[n]) * d(ac[n])/dx_j / s^2 - 2*|ac[n]|/s^3
    # So d(f)/dx_j = sum_n exp(...) * beta * [sign(ac[n]) * d(ac[n])/dx_j / s^2 - 2*|ac[n]|/s^3]
    
    # Simplify: weights[n] = exp(beta * (|ac[n]|/s^2 - t))
    weights = exp_shifted
    g1 = weights * np.sign(ac) / s2  # for the sign term
    g2 = weights * abs_ac / (s2 * s2)  # for the -2|ac|/s^3 term, times 2
    
    # d(ac[n])/dx_j = x[n-j] + x[j-n]
    # So sum_n g1[n] * (x[n-j] + x[j-n]) and sum_n g2[n] * (x[n-j] + x[j-n])
    
    L_fft = 1
    while L_fft < 2 * N - 1:
        L_fft *= 2
    
    G1 = np.fft.rfft(g1, n=L_fft)
    G2 = np.fft.rfft(g2, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1_1 = np.fft.irfft(np.conj(G1) * Xf, n=L_fft)[:N]
    term2_1 = np.fft.irfft(G1 * Xf, n=L_fft)[:N]
    term1_2 = np.fft.irfft(np.conj(G2) * Xf, n=L_fft)[:N]
    term2_2 = np.fft.irfft(G2 * Xf, n=L_fft)[:N]
    
    d_penalty_dx = (1.0 / s2) * beta * (term1_1 + term2_1 - 2.0 * (term1_2 + term2_2))
    
    # Gradient w.r.t. t:
    # d/dt [t + penalty/s^2] = 1 + (1/s^2) * sum_n exp(...) * (-beta)
    d_val_dt = 1.0 - beta * np.sum(exp_shifted) / s2
    
    grad = np.concatenate([d_penalty_dx, [d_val_dt]])
    
    return val, grad

def _coordinate_descent(x, max_iters=5000, step_scale=0.1):
    """
    Coordinate descent polish: adjust individual coordinates to reduce objective.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            # Try small adjustments to x[j]
            for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                trial = best_x.copy()
                trial[j] += delta
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_x, best_val

def _spectral_design(a, tilt=0.0, mod=0.0):
    """
    Generate a step function from spectral parameters.
    a: shape parameter for sinc-like spectrum
    tilt: frequency tilt
    mod: modulation amount
    """
    a = np.clip(a, 0.5, 3.0)
    tilt = np.clip(tilt, -1.0, 1.0)
    mod = np.clip(mod, 0.0, 1.0)
    
    k = np.arange(N)
    c = 0.5 + a
    kf = k / N
    
    mag = np.exp(-c * kf**2)
    mag = mag * np.exp(tilt * kf)
    
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    if mod > 0.01:
        t = np.linspace(0, 1, N)
        x = x * (1.0 + mod * 0.5 * np.sin(2*np.pi*t))
        x = x / np.max(np.abs(x))
    
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.exp(-2.0 * t),
        np.exp(2.0 * t - 2.0),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 5 * t),
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    # Add some random smooth initializations
    rng_init = np.random.default_rng(123)
    for _ in range(20):
        freqs = rng_init.uniform(0.1, 3.0, size=5)
        phases = rng_init.uniform(0, 2*np.pi, size=5)
        amps = rng_init.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate
    def _smooth_obj(x):
        val, grad = _objective_grad_smooth(x, beta=30.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Try multiple beta values for soft-max
    for beta in [10.0, 20.0, 50.0, 100.0]:
        if _check_deadline():
            break
        
        def _obj_beta(x, b=beta):
            val, grad = _objective_grad_smooth(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_beta, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 3b: Epigraph formulation optimization
    if not _check_deadline():
        try:
            ac_init = _autoconv_fft(best_x)
            s_init = np.sum(best_x)
            t_init = np.max(np.abs(ac_init)) / (s_init * s_init)
            params_init = np.concatenate([best_x, [t_init]])
            
            result = minimize(_epigraph_objective, params_init, method='L-BFGS-B', jac=True,
                            options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-12})
            
            if result.fun < 1e9:
                x_ep = result.x[:N]
                val_ep = _objective(x_ep)
                if val_ep < best_val:
                    best_val = val_ep
                    best_x = x_ep.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Spectral parameter optimization
    def _param_objective(params):
        x = _spectral_design(params[0], params[1], params[2])
        return _objective(x)
    
    param_initials = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 0.5, 0.0],
        [1.5, -0.5, 0.0],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]
    
    for p0 in param_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_param_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _spectral_design(result.x[0], result.x[1], result.x[2])
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=3000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Final L-BFGS-B polish
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Peak-targeted polish
    # Identify dominant autoconvolution peaks and try to reduce them specifically
    rng = np.random.default_rng(42)
    ac_current = _autoconv_fft(best_x)
    s_current = np.sum(best_x)
    abs_ac_current = np.abs(ac_current)
    max_ac_val = np.max(abs_ac_current)
    
    # Find top K peaks
    K = min(20, 2*N-1)
    peak_indices = np.argsort(abs_ac_current)[-K:]
    
    # For each peak, try local adjustments to reduce it
    for peak_idx in peak_indices:
        if _check_deadline():
            break
        
        # The autoconvolution at lag peak_idx is sum_j x[j]*x[peak_idx-j]
        # To reduce it, we can try flipping signs or adjusting magnitudes
        # of x[j] where x[j]*x[peak_idx-j] contributes positively to the peak
        
        # Get contributing pairs
        j_range = np.arange(max(0, peak_idx - N + 1), min(N, peak_idx + 1))
        if len(j_range) == 0:
            continue
        
        partner_indices = peak_idx - j_range
        partner_indices = partner_indices[(partner_indices >= 0) & (partner_indices < N)]
        j_range = j_range[(partner_indices >= 0) & (partner_indices < N)]
        
        if len(j_range) == 0:
            continue
        
        # Try small adjustments to reduce the peak
        for j in j_range[:min(50, len(j_range))]:  # Limit to 50 trials per peak
            if _check_deadline():
                break
            
            for delta in [-0.1, -0.05, 0.05, 0.1]:
                trial = best_x.copy()
                trial[j] += delta
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    break
    
    # Phase 7b: Random perturbation refinement with adaptive step
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    max_iters = 50000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        # Use smooth surrogate for fast evaluation
        val_smooth, _ = _objective_grad_smooth(x_trial, beta=50.0)
        
        if val_smooth < best_val * 0.999:  # Require clear improvement in surrogate
            val_exact = _objective(x_trial)
            if val_exact < best_val:
                best_val = val_exact
                best_x = x_trial.copy()
                stall_count = 0
                step_size = max(0.001, step_size * 0.9998)
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        if stall_count % 5000 == 0:
            step_size = min(0.5, step_size * 1.05)
            if stall_count % 20000 == 0:
                big_pert = rng.normal(0, step_size * 5, size=N)
                x_big = best_x + big_pert
                val_big = _objective(x_big)
                if val_big < best_val:
                    best_val = val_big
                    best_x = x_big.copy()
                    stall_count = 0
    
    # Phase 8: Epigraph optimization for final refinement
    if not _check_deadline():
        try:
            # Initialize t to current max|ac|/s^2
            ac_init = _autoconv_fft(best_x)
            s_init = np.sum(best_x)
            t_init = np.max(np.abs(ac_init)) / (s_init * s_init)
            
            params_init = np.concatenate([best_x, [t_init]])
            
            result = minimize(_epigraph_objective, params_init, method='L-BFGS-B', jac=True,
                            options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-12})
            
            if result.fun < 1e9:
                x_ep = result.x[:N]
                val_ep = _objective(x_ep)
                if val_ep < best_val:
                    best_val = val_ep
                    best_x = x_ep.copy()
        except Exception:
            pass
    
    # Phase 9: High-beta L-BFGS polish with tight tolerance
    if not _check_deadline():
        try:
            def _high_beta_obj(x):
                val, grad = _objective_grad_smooth(x, beta=100.0)
                return val, grad
            
            result = minimize(_high_beta_obj, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish: tight BFGS on exact objective
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
