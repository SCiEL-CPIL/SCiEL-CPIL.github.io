"""
Spectral alternating minimization for signed step functions minimizing max|autoconv|/(sum x)^2.
Alternates between: (1) optimizing phase in spectral domain with soft magnitude constraint,
(2) projecting back to time domain with soft spectral magnitude regularization.
"""

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

def _fft_len():
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    return L, L_fft

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L, L_fft = _fft_len()
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max|autoconv| / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_phase_objective_and_grad(phases, target_mag, beta):
    """
    Objective for phase optimization in spectral domain.
    phases: array of shape (n_pos,) with phases[0] fixed to 0
    target_mag: desired magnitude spectrum
    beta: soft constraint strength for magnitude deviation
    
    Objective = max|autoconv|/(sum x)^2 + beta * |||X| - target_mag||^2 / (sum x)^2
    """
    n_pos = len(phases)
    L, L_fft = _fft_len()
    
    # Build complex spectrum
    X = np.zeros(n_pos, dtype=complex)
    for k in range(n_pos):
        X[k] = target_mag[k] * np.exp(1j * phases[k])
    
    # Convert to time domain
    x = np.fft.irfft(X, n=L_fft)[:N]
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(n_pos)
    
    # Compute autoconvolution
    ac = np.fft.irfft(X * X, n=L_fft)[:L]
    abs_ac = np.abs(ac)
    
    # Smooth max using LogSumExp
    mu = 0.1
    max_abs = np.max(abs_ac)
    if max_abs < 1e-12:
        M_val = 1e-12
    else:
        shifted = (abs_ac - max_abs) / mu
        exp_terms = np.exp(shifted)
        sum_exp = np.sum(exp_terms)
        log_sum_exp = max_abs / mu + np.log(sum_exp)
        M_val = mu * log_sum_exp
    
    base_val = M_val / (s * s)
    
    # Magnitude penalty
    mag = np.abs(X)
    mag_dev = mag - target_mag
    mag_penalty = np.sum(mag_dev ** 2) / (s * s)
    
    total_val = base_val + beta * mag_penalty
    
    # Gradient with respect to phases
    # d/dphases[k] of X[k] = i * target_mag[k] * exp(i*phases[k]) = i * X[k]
    # d/dphases[k] of X*[k] = -i * X*[k]
    
    # Gradient of M_val w.r.t. phases
    # M_val depends on X through ac = ifft(X*X)
    # d(M_val)/dphases[k] = Re( conj(dac/dX[k]) * dM/dac * dX/dphases[k] )
    # This is complex; use numerical or FFT-based approach
    
    # Simpler: use the weight approach
    weights = np.sign(ac) * exp_terms / sum_exp
    
    # grad_M[j] = sum_k weights[k] (x[j+k] + x[j-k])
    W_fft = np.fft.rfft(weights, n=L_fft)
    X_fft = np.fft.rfft(x, n=L_fft)
    
    conv1 = np.fft.irfft(W_fft * X_fft, n=L_fft)[:L]
    conv2 = np.fft.irfft(np.conj(W_fft) * X_fft, n=L_fft)[:L]
    grad_M = (conv1 + conv2)[:N]
    
    # d(M_val)/dphases[k] via chain rule through X
    # M_val = mu * log(sum(exp(|ac|/mu)))
    # dM/dac[l] = sign(ac[l]) * exp((|ac[l]|-max_abs)/mu) / sum_exp = weights[l]
    # dac[l]/dX[m] = X[m] if l == m, else 0 (since ac = ifft(X*X), ac[l] = (1/L_fft) sum_m X[m]*X[l-m])
    # Actually ac[l] = (1/L_fft) * sum_m X[m] * X[l-m]
    # d(ac[l])/d(X[m]) = (1/L_fft) * (X[l-m] + X[l-m] if l-m != m else 2*X[m])
    # This is getting complex. Let's use a different approach.
    
    # Alternative: compute gradient numerically or use a simpler formulation
    # For now, return the base gradient and handle magnitude penalty separately
    
    # Gradient of magnitude penalty w.r.t. phases
    # d(mag_penalty)/dphases[k] = 2 * (mag[k] - target_mag[k]) * d(mag[k])/dphases[k]
    # mag[k] = |X[k]| = target_mag[k] (fixed!), so this is 0 if we fix magnitudes
    # But we're allowing soft deviation, so we need to be careful
    
    # Actually, in this formulation, X[k] = target_mag[k] * exp(i*phases[k])
    # So |X[k]| = target_mag[k] exactly, and the magnitude penalty is always 0
    # This means the beta term is redundant in this parameterization
    
    # Let's simplify: just optimize phases with fixed magnitudes
    # Gradient of base_val w.r.t. phases[k]
    
    # Use the fact that x = ifft(X) and ac = ifft(X*X)
    # d(base_val)/dphases[k] = (1/s^2) * d(M_val)/dphases[k] - 2*M_val/s^3 * d(s)/dphases[k]
    
    # d(s)/dphases[k] = Re(d(ifft(X))/dphases[k] summed)
    # d(x[j])/dphases[k] = Re(i * X[k] * exp(-i*2*pi*k*j/L_fft)) / L_fft
    # d(s)/dphases[k] = Re(i * X[k] * sum_j exp(-i*2*pi*k*j/L_fft)) / L_fft
    # sum_j exp(-i*2*pi*k*j/L_fft) = L_fft if k==0 else 0 (for j=0..N-1, not full FFT)
    # This is tricky because we only have N samples, not L_fft
    
    # Let's use a practical approach: finite differences for the phase gradient
    # Or better: reformulate to avoid the complexity
    
    # For now, return a simple gradient estimate
    grad = np.zeros(n_pos)
    
    # Numerical gradient for phases
    eps = 1e-5
    for k in range(1, n_pos):  # Skip k=0 (DC, phase fixed)
        phases_plus = phases.copy()
        phases_plus[k] += eps
        X_plus = np.zeros(n_pos, dtype=complex)
        for m in range(n_pos):
            X_plus[m] = target_mag[m] * np.exp(1j * phases_plus[m])
        x_plus = np.fft.irfft(X_plus, n=L_fft)[:N]
        s_plus = np.sum(x_plus)
        if abs(s_plus) < 1e-10:
            grad[k] = 0
            continue
        ac_plus = np.fft.irfft(X_plus * X_plus, n=L_fft)[:L]
        val_plus = np.max(np.abs(ac_plus)) / (s_plus * s_plus)
        
        phases_minus = phases.copy()
        phases_minus[k] -= eps
        X_minus = np.zeros(n_pos, dtype=complex)
        for m in range(n_pos):
            X_minus[m] = target_mag[m] * np.exp(1j * phases_minus[m])
        x_minus = np.fft.irfft(X_minus, n=L_fft)[:N]
        s_minus = np.sum(x_minus)
        if abs(s_minus) < 1e-10:
            grad[k] = 0
            continue
        ac_minus = np.fft.irfft(X_minus * X_minus, n=L_fft)[:L]
        val_minus = np.max(np.abs(ac_minus)) / (s_minus * s_minus)
        
        grad[k] = (val_plus - val_minus) / (2 * eps)
    
    return total_val, grad

def _spectral_alternating_min(x_init, target_mag, max_sweeps=50, beta_schedule=None):
    """
    Alternating minimization: optimize phases in spectral domain, project to time domain.
    """
    n_pos = len(target_mag)
    L, L_fft = _fft_len()
    
    # Initialize phases
    rng = np.random.RandomState(42)
    phases = rng.uniform(0, 2 * np.pi, n_pos)
    phases[0] = 0.0  # Fix DC phase
    
    x_best = x_init.copy()
    best_val = _objective(x_best)
    
    # Beta schedule: start with strong magnitude constraint, weaken over time
    if beta_schedule is None:
        beta_schedule = np.geomspace(1.0, 0.01, max_sweeps)
    
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        
        beta = beta_schedule[sweep] if sweep < len(beta_schedule) else 0.01
        
        # Optimize phases
        def phase_obj(p):
            if _check_deadline():
                return 1e10, np.zeros(n_pos)
            val, grad = _spectral_phase_objective_and_grad(p, target_mag, beta)
            return val, grad
        
        try:
            result = minimize(
                phase_obj,
                phases,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0, 2*np.pi)] * n_pos,
                options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 30}
            )
            
            if result.fun < 1e9:
                phases = result.x.copy()
                phases[0] = 0.0
                
                # Convert to time domain
                X = np.zeros(n_pos, dtype=complex)
                for k in range(n_pos):
                    X[k] = target_mag[k] * np.exp(1j * phases[k])
                x_new = np.fft.irfft(X, n=L_fft)[:N]
                
                # Normalize
                s_new = np.sum(x_new)
                if abs(s_new) > 1e-10:
                    x_new = x_new / s_new
                    val_new = _objective(x_new)
                    if val_new < best_val:
                        best_val = val_new
                        x_best = x_new.copy()
        except Exception:
            pass
    
    return x_best, best_val

def _design_flat_mag_spectrum():
    """Design a flat magnitude spectrum for optimal phase retrieval."""
    n_pos = L_fft // 2 + 1
    mag = np.ones(n_pos)
    # Normalize so that sum of x is 1 (DC component)
    mag[0] = 1.0
    # Scale other components
    mag[1:] = 1.0 / np.sqrt(n_pos - 1)
    return mag

def _coordinate_descent_polish(x, max_iters=15):
    """
    Final polish using coordinate descent on the true objective.
    """
    best_x = x.copy()
    s = np.sum(best_x)
    best_val = _objective(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            if _check_deadline():
                break
            for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                trial = best_x.copy()
                trial[j] += delta
                s_t = np.sum(trial)
                if abs(s_t) < 1e-10:
                    continue
                val_t = _objective(trial)
                if val_t < best_val:
                    best_val = val_t
                    best_x = trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_x, best_val

def construct():
    global L_fft
    L, L_fft = _fft_len()
    n_pos = L_fft // 2 + 1
    
    rng = np.random.RandomState(42)
    
    best_x = None
    best_val = 1e10
    
    # Design target magnitude spectrum
    target_mag = _design_flat_mag_spectrum()
    
    # Try multiple random phase initializations
    n_trials = 5
    
    for trial in range(n_trials):
        if _check_deadline():
            break
        
        # Random phase initialization
        phases = rng.uniform(0, 2 * np.pi, n_pos)
        phases[0] = 0.0
        
        X = target_mag * np.exp(1j * phases)
        x_init = np.fft.irfft(X, n=L_fft)[:N]
        
        # Normalize
        s_init = np.sum(x_init)
        if abs(s_init) < 1e-10:
            continue
        x_init = x_init / s_init
        
        # Run alternating minimization
        x_opt, val_opt = _spectral_alternating_min(x_init, target_mag, max_sweeps=40)
        
        # Polish with coordinate descent
        x_pol, val_pol = _coordinate_descent_polish(x_opt, max_iters=10)
        
        if val_pol < best_val:
            best_val = val_pol
            best_x = x_pol.copy()
    
    # Also try a few time-domain random initializations with spectral projection
    for trial in range(3):
        if _check_deadline():
            break
        
        x_rand = rng.randn(N)
        s_rand = np.sum(x_rand)
        if abs(s_rand) < 1e-10:
            continue
        x_rand = x_rand / s_rand
        
        # Project to flat spectral magnitude
        X = np.fft.rfft(x_rand, n=L_fft)
        X_new = target_mag * np.exp(1j * np.angle(X))
        X_new[0] = target_mag[0]  # Keep DC real and positive
        x_proj = np.fft.irfft(X_new, n=L_fft)[:N]
        s_proj = np.sum(x_proj)
        if abs(s_proj) < 1e-10:
            continue
        x_proj = x_proj / s_proj
        
        # Optimize from this point
        x_opt, val_opt = _spectral_alternating_min(x_proj, target_mag, max_sweeps=30)
        x_pol, val_pol = _coordinate_descent_polish(x_opt, max_iters=10)
        
        if val_pol < best_val:
            best_val = val_pol
            best_x = x_pol.copy()
    
    if best_x is None:
        best_x = np.ones(N) / N
    
    return best_x
# EVOLVE-BLOCK-END
