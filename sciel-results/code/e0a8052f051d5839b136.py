"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_design(target_flatness=1.0, n_freqs=None):
    """
    Design x in frequency domain to have a flat |X(omega)|^2 profile.
    This directly minimizes peak autoconvolution since FFT(autoconv) = |X|^2.
    """
    if n_freqs is None:
        n_freqs = N
    
    # Target: |X(omega)|^2 should be as flat as possible
    # We design X(omega) = A * exp(i*phi(omega)) where A is constant magnitude
    # and phi is chosen to make x real
    
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    # Number of positive frequency bins
    n_pos = L_fft // 2 + 1
    
    # Design target power spectrum: flat with some low-frequency emphasis
    # to ensure x has reasonable DC component
    freqs = np.fft.rfftfreq(L_fft, d=1.0)
    
    # Flat spectrum with gentle roll-off at high frequencies
    # This helps keep x smooth
    target_power = np.ones(n_pos)
    # Gentle low-pass to keep x smooth
    for i in range(n_pos):
        if freqs[i] > 0.1:
            target_power[i] = 1.0 / (1.0 + 2.0 * freqs[i])
    
    # Normalize so that sum(x) is reasonable
    # sum(x) = X[0] (DC component)
    # We want |X[0]|^2 to be large relative to other bins
    
    # Design X with constant magnitude per bin, scaled by target_power
    X_target = np.sqrt(target_power) * target_flatness
    
    # Ensure DC is dominant for good normalization
    X_target[0] = np.sqrt(N) * target_flatness
    
    # Choose phases to make x real: X[-k] = conj(X[k])
    # For rfft, we just need to ensure the inverse is real
    # Random phases work well
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, size=n_pos)
    phases[0] = 0  # DC is real
    
    X_complex = X_target * np.exp(1j * phases)
    
    # Inverse FFT to get x
    x = np.fft.irfft(X_complex, n=L_fft)[:N]
    
    # Ensure positive sum (flip sign if needed)
    if np.sum(x) < 0:
        x = -x
    
    return x

def _peak_targeted_polish(x, max_iters=5000, n_top_peaks=5):
    """
    Subgradient-style polish that directly targets the top autoconvolution peaks.
    At each step, find the lag with maximum |autoconv|, and move x in the
    direction that reduces that specific peak.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    L_fft = 1
    while L_fft < 3 * N:
        L_fft *= 2
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        s = np.sum(best_x)
        if abs(s) < 1e-10:
            break
        
        # Find top peaks
        peak_indices = np.argsort(abs_ac)[-n_top_peaks:]
        
        # For each peak, compute the subgradient direction
        # d|ac[n]|/dx[j] = sign(ac[n]) * (x[j+n] + x[j-n])
        # We want to decrease the peak, so move opposite to subgradient
        
        # Weight peaks by their magnitude
        weights = abs_ac[peak_indices]
        total_w = np.sum(weights)
        if total_w < 1e-15:
            break
        
        # Compute aggregate subgradient
        grad = np.zeros(N)
        for idx, w in zip(peak_indices, weights):
            n = idx
            sign_ac = np.sign(ac[idx])
            # d(ac[n])/dx[j] = x[j+n] + x[j-n]
            # Only valid when indices are in range
            for j in range(N):
                jn = j + n
                jm = j - n
                contrib = 0.0
                if 0 <= jn < N:
                    contrib += best_x[jn]
                if 0 <= jm < N:
                    contrib += best_x[jm]
                grad[j] += sign_ac * contrib * (w / total_w)
        
        # Normalize gradient
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-15:
            break
        
        grad_norm = grad / g_norm
        
        # Adaptive step size
        step_size = 0.01 * (1.0 + 0.5 * np.sin(0.01 * it))
        
        # Try moving in negative gradient direction
        x_trial = best_x - step_size * grad_norm
        
        val_trial = _objective(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
        else:
            # Try smaller step
            step_size *= 0.5
            x_trial = best_x - step_size * grad_norm
            val_trial = _objective(x_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_x = x_trial.copy()
            else:
                # Try positive direction (maybe we're at a local min for this peak)
                x_trial = best_x + step_size * 0.5 * grad_norm
                val_trial = _objective(x_trial)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = x_trial.copy()
    
    return best_x, best_val

def _coordinate_descent(x, max_iters=5000, step_scale=0.1):
    """
    Coordinate descent polish: adjust individual coordinates to reduce objective.
    Uses in-place modification for speed.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            orig_val = best_x[j]
            for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                best_x[j] = orig_val + delta
                val = _objective(best_x)
                if val < best_val:
                    best_val = val
                    improved = True
                    break
            else:
                best_x[j] = orig_val
        
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Spectral design initializations
    # The key insight: FFT(autoconv) = |X|^2, so flat |X|^2 minimizes peak autoconv
    spectral_inits = []
    
    for flatness in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        x_spec = _spectral_design(target_flatness=flatness)
        spectral_inits.append(x_spec)
    
    # Also try with different phase seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if _check_deadline():
            break
        # Modify _spectral_design to accept seed
        L_fft = 1
        while L_fft < 2 * N:
            L_fft *= 2
        n_pos = L_fft // 2 + 1
        freqs = np.fft.rfftfreq(L_fft, d=1.0)
        target_power = np.ones(n_pos)
        for i in range(n_pos):
            if freqs[i] > 0.1:
                target_power[i] = 1.0 / (1.0 + 2.0 * freqs[i])
        X_target = np.sqrt(target_power) * 1.0
        X_target[0] = np.sqrt(N)
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, 2 * np.pi, size=n_pos)
        phases[0] = 0
        X_complex = X_target * np.exp(1j * phases)
        x = np.fft.irfft(X_complex, n=L_fft)[:N]
        if np.sum(x) < 0:
            x = -x
        spectral_inits.append(x)
    
    for x0 in spectral_inits:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # Phase 2: Structured initializations (keep some for diversity)
    t = np.linspace(0, 1, N)
    structured_inits = [
        np.ones(N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        np.exp(-2.0 * t),
        1.0 + 0.3 * np.sin(np.pi * t),
    ]
    
    for x0 in structured_inits:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: L-BFGS-B with smooth surrogate from best initial
    def _smooth_obj(x, beta=30.0):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(x)
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        if max_val < 1e-15:
            return 1e10, np.zeros_like(x)
        
        shifted = beta * (abs_ac - max_val)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
        val = soft_max / (s * s)
        
        g = np.zeros(2*N-1)
        g[:len(ac)] = weights * np.sign(ac)
        
        L_fft = 1
        while L_fft < 3*N:
            L_fft *= 2
        
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(x, n=L_fft)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        
        d_softmax_dx = term1 + term2
        
        grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
        
        return val, grad
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Peak-targeted subgradient polish (the key structural addition)
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=3000, n_top_peaks=3)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: More L-BFGS-B refinement
    for beta in [50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        
        def _obj_beta(x, b=beta):
            return _smooth_obj(x, beta=b)
        
        try:
            result = minimize(_obj_beta, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=3000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Another round of peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=2000, n_top_peaks=5)
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Final coordinate descent with smaller steps
    best_x, best_val = _coordinate_descent(best_x, max_iters=2000, step_scale=0.02)
    
    return best_x
# EVOLVE-BLOCK-END
