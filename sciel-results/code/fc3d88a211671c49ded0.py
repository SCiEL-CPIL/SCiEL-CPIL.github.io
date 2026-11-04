"""Direct phase optimization in frequency domain: minimize peak-to-mean ratio of autocorrelation."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _peak_to_mean_ratio(conv):
    """Compute peak-to-mean power ratio of autocorrelation (excluding DC)."""
    abs_conv = np.abs(conv)
    # Exclude DC term (lag 0)
    if len(abs_conv) > 1:
        offdiag_mean = np.mean(abs_conv[1:])
        peak = np.max(abs_conv)
    else:
        offdiag_mean = 1e-10
        peak = np.max(abs_conv)
    if offdiag_mean < 1e-12:
        return peak / 1e-12
    return peak / offdiag_mean


def _phase_objective(phases, mags, target_dc, beta=200.0):
    """
    Objective: minimize the peak-to-mean ratio of autocorrelation.
    We optimize only the phases of the FFT coefficients, keeping magnitudes fixed.
    This directly targets spectral flatness of the autocorrelation.
    """
    n_freqs = len(mags)
    fx = np.zeros(n_freqs, dtype=complex)
    fx[0] = target_dc  # DC component fixed to maintain integral
    for k in range(1, n_freqs):
        fx[k] = mags[k] * np.exp(1j * phases[k - 1])
    
    x = np.fft.irfft(fx, _FFT_SIZE)[:N]
    s = _integral(x)
    
    if abs(s) < 1e-12:
        return 1e10
    
    conv = _autoconv_values(x)
    abs_conv = np.abs(conv)
    
    # Soft max for smoothness
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    soft_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # Mean of off-diagonal terms
    if len(abs_conv) > 1:
        offdiag_mean = np.mean(abs_conv[1:])
    else:
        offdiag_mean = 1e-10
    
    if offdiag_mean < 1e-12:
        return soft_max / 1e-12
    
    # Peak-to-mean ratio, normalized by s^2
    p2m = soft_max / (offdiag_mean * s * s)
    
    # Also penalize if integral deviates from target
    dc_penalty = 10.0 * (s - target_dc) ** 2
    
    return p2m + dc_penalty


def _phase_gradient(phases, mags, target_dc, beta=200.0):
    """Finite difference gradient for phase optimization."""
    n_phases = len(phases)
    eps = 1e-4
    grad = np.zeros(n_phases)
    
    for j in range(n_phases):
        phases_plus = phases.copy()
        phases_plus[j] += eps
        val_plus = _phase_objective(phases_plus, mags, target_dc, beta)
        
        phases_minus = phases.copy()
        phases_minus[j] -= eps
        val_minus = _phase_objective(phases_minus, mags, target_dc, beta)
        
        grad[j] = (val_plus - val_minus) / (2 * eps)
    
    return grad


def _optimize_phases(x_init, deadline, max_iters=3000, beta=200.0):
    """
    Optimize phases of FFT to minimize peak-to-mean autocorrelation ratio.
    This is a direct frequency-domain approach that targets spectral flatness.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    fx = np.fft.rfft(best_x, _FFT_SIZE)
    mags = np.abs(fx).copy()
    mags[mags < 1e-10] = 1e-10
    
    target_dc = _integral(best_x)
    
    n_freqs = len(fx)
    n_phases = n_freqs - 1
    
    # Initialize phases from current signal
    phases_init = np.zeros(n_phases)
    for k in range(1, n_freqs):
        if mags[k] > 1e-10:
            phases_init[k - 1] = np.angle(fx[k])
    
    def objective(params):
        return _phase_objective(params, mags, target_dc, beta)
    
    def gradient(params):
        return _phase_gradient(params, mags, target_dc, beta)
    
    time_limit = max(1.0, deadline - time.time() - 5.0)
    
    try:
        result = minimize(
            objective,
            phases_init,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-6,
                     'maxfun': 50000, 'maxls': 50}
        )
        
        # Reconstruct from optimized phases
        phases_opt = result.x[:n_phases]
        fx_final = np.zeros(n_freqs, dtype=complex)
        fx_final[0] = target_dc
        for k in range(1, n_freqs):
            fx_final[k] = mags[k] * np.exp(1j * phases_opt[k - 1])
        
        x_final = np.fft.irfft(fx_final, _FFT_SIZE)[:N]
        val_final = _metric(x_final)
        
        if val_final < best_val:
            best_val = val_final
            best_x = x_final.copy()
    except Exception:
        pass
    
    return best_x, best_val


def _multi_phase_restart(x_init, deadline, rng, n_restarts=5):
    """
    Multiple restarts with random phase perturbations to escape local minima.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    fx = np.fft.rfft(best_x, _FFT_SIZE)
    mags = np.abs(fx).copy()
    mags[mags < 1e-10] = 1e-10
    target_dc = _integral(best_x)
    
    n_freqs = len(fx)
    n_phases = n_freqs - 1
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random phase perturbation
        if restart == 0:
            phases_init = np.zeros(n_phases)
            for k in range(1, n_freqs):
                if mags[k] > 1e-10:
                    phases_init[k - 1] = np.angle(fx[k])
        else:
            # Add random perturbation to phases
            phases_init = np.zeros(n_phases)
            for k in range(1, n_freqs):
                if mags[k] > 1e-10:
                    phases_init[k - 1] = np.angle(fx[k]) + rng.normal(0, 0.5, 1)[0]
        
        def objective(params):
            return _phase_objective(params, mags, target_dc, beta=150.0)
        
        def gradient(params):
            return _phase_gradient(params, mags, target_dc, beta=150.0)
        
        try:
            result = minimize(
                objective,
                phases_init,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-6,
                         'maxfun': 30000, 'maxls': 50}
            )
            
            phases_opt = result.x[:n_phases]
            fx_final = np.zeros(n_freqs, dtype=complex)
            fx_final[0] = target_dc
            for k in range(1, n_freqs):
                fx_final[k] = mags[k] * np.exp(1j * phases_opt[k - 1])
            
            x_final = np.fft.irfft(fx_final, _FFT_SIZE)[:N]
            val_final = _metric(x_final)
            
            if val_final < best_val:
                best_val = val_final
                best_x = x_final.copy()
        except Exception:
            pass
    
    return best_x, best_val


def _soft_max_abs(conv, beta=1000.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_metric(x, beta=500.0):
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)


def _gradient_descent_smooth(x_init, deadline, rng, max_iters=15000):
    """Gradient descent using smooth log-sum-exp surrogate."""
    best_x = x_init.copy()
    best_val = _metric(best_x)
    current_x = x_init.copy()
    
    lr = 0.003
    patience = 0
    best_smooth_val = _smooth_metric(current_x, beta=300.0)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        conv = _autoconv_values(current_x)
        abs_conv = np.abs(conv)
        s = _integral(current_x)
        
        if abs(s) < 1e-12:
            break
        
        beta = 300.0
        shifted = abs_conv - np.max(abs_conv)
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        grad = np.zeros(N)
        for k in range(2 * N - 1):
            if weights[k] < 1e-10:
                continue
            sign_k = np.sign(conv[k]) if conv[k] != 0 else 1.0
            for j in range(N):
                contrib = 0.0
                if j + k < N:
                    contrib += current_x[j + k]
                if j - k >= 0:
                    contrib += current_x[j - k]
                grad[j] += 2.0 * sign_k * weights[k] * contrib
        
        soft_max_val = _soft_max_abs(conv, beta)
        grad *= -2.0 * soft_max_val / (s ** 3)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        x_new = current_x - lr * grad
        v_new = _metric(x_new)
        if v_new < best_val:
            best_val = v_new
            best_x = x_new.copy()
            current_x = x_new
            patience = 0
            lr = min(lr * 1.05, 0.05)
        else:
            patience += 1
            if patience > 30:
                lr *= 0.7
                patience = 0
                if lr < 1e-7:
                    break
    
    return best_x, best_val


def _local_search_fast(x, deadline, rng, max_rounds=80000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 150:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.05, 0.05
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(8):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Direct phase optimization (primary new strategy)
    if time.time() < deadline - 60:
        x_phase, val_phase = _optimize_phases(best_x, deadline - 60, max_iters=2500, beta=200.0)
        if val_phase < best_val:
            best_val = val_phase
            best_x = x_phase
    
    # Phase 2: Multi-restart phase optimization
    if time.time() < deadline - 35:
        x_multi, val_multi = _multi_phase_restart(best_x, deadline - 35, rng, n_restarts=4)
        if val_multi < best_val:
            best_val = val_multi
            best_x = x_multi
    
    # Phase 3: Gradient descent with smooth surrogate
    if time.time() < deadline - 20:
        x_grad, val_grad = _gradient_descent_smooth(best_x, deadline - 20, rng, max_iters=8000)
        if val_grad < best_val:
            best_val = val_grad
            best_x = x_grad
    
    # Phase 4: Local search
    if time.time() < deadline - 10:
        best_x, best_val = _local_search_fast(best_x, deadline - 10, rng, max_rounds=40000)
    
    # Phase 5: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
