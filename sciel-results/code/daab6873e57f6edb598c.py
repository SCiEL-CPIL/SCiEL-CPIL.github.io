import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _chirp_heights(alpha, phase_offset=0.0):
    """
    Generate step heights from a complex chirp with quadratic phase.
    h_i = |exp(i * (2*pi*i/N + alpha * i^2/N)) + exp(i * phase_offset)|^2 * scale
    
    The chirp structure inherently has low autocorrelation.
    We normalize to sum to N/2 and clip to [0,1].
    """
    i = np.arange(N, dtype=np.float64)
    # Quadratic phase: alpha * i^2 / N creates the chirp
    phase = 2.0 * np.pi * i / N + alpha * i**2 / N + phase_offset
    # Complex signal
    z = np.exp(1j * phase)
    # Squared magnitude gives heights
    h = np.abs(z + 1.0)**2  # Add DC component to ensure positive values
    # Normalize to sum N/2
    h = h / np.sum(h) * (N / 2.0)
    # Clip to [0,1]
    h = np.clip(h, 0.0, 1.0)
    # Re-normalize after clipping
    diff = N/2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h


def _chirp_smooth_max(alpha, phase_offset, beta):
    """
    Smooth max of overlap values for a chirp parameterization.
    Returns (value, gradient w.r.t. alpha and phase_offset)
    """
    h = _chirp_heights(alpha, phase_offset)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Numerical gradients for chirp parameters
    eps = 1e-7
    h_alpha_plus = _chirp_heights(alpha + eps, phase_offset)
    cs_alpha_plus = _linear_overlap_values(h_alpha_plus)
    val_alpha_plus = np.max(cs_alpha_plus) + np.log(np.sum(np.exp(beta * (cs_alpha_plus - np.max(cs_alpha_plus))))) / beta
    
    h_alpha_minus = _chirp_heights(alpha - eps, phase_offset)
    cs_alpha_minus = _linear_overlap_values(h_alpha_minus)
    val_alpha_minus = np.max(cs_alpha_minus) + np.log(np.sum(np.exp(beta * (cs_alpha_minus - np.max(cs_alpha_minus))))) / beta
    
    grad_alpha = (val_alpha_plus - val_alpha_minus) / (2 * eps)
    
    h_phase_plus = _chirp_heights(alpha, phase_offset + eps)
    cs_phase_plus = _linear_overlap_values(h_phase_plus)
    val_phase_plus = np.max(cs_phase_plus) + np.log(np.sum(np.exp(beta * (cs_phase_plus - np.max(cs_phase_plus))))) / beta
    
    h_phase_minus = _chirp_heights(alpha, phase_offset - eps)
    cs_phase_minus = _linear_overlap_values(h_phase_minus)
    val_phase_minus = np.max(cs_phase_minus) + np.log(np.sum(np.exp(beta * (cs_phase_minus - np.max(cs_phase_minus))))) / beta
    
    grad_phase = (val_phase_plus - val_phase_minus) / (2 * eps)
    
    return val, np.array([grad_alpha, grad_phase])


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 1: Search over chirp parameters
    # Coarse grid search over alpha and phase_offset
    alpha_range = np.linspace(-5.0, 5.0, 50)
    phase_range = np.linspace(0.0, 2.0 * np.pi, 20)
    
    for alpha in alpha_range:
        if time.time() - t0 > _DEADLINE - 60:
            break
        for phase in phase_range:
            h = _chirp_heights(alpha, phase)
            _update_best(h)
    
    # Phase 2: Refine best chirp parameters with gradient-based optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        # Find the alpha and phase that gave the best score
        # Re-search more finely around the best region
        best_alpha = None
        best_phase = None
        best_chirp_score = float('inf')
        
        alpha_range_fine = np.linspace(-5.0, 5.0, 100)
        phase_range_fine = np.linspace(0.0, 2.0 * np.pi, 40)
        
        for alpha in alpha_range_fine:
            if time.time() - t0 > _DEADLINE - 40:
                break
            for phase in phase_range_fine:
                h = _chirp_heights(alpha, phase)
                s = _score(h)
                if s < best_chirp_score:
                    best_chirp_score = s
                    best_alpha = alpha
                    best_phase = phase
        
        # Gradient-based refinement of chirp parameters
        if best_alpha is not None:
            def chirp_obj(params):
                val, _ = _chirp_smooth_max(params[0], params[1], beta=50.0)
                return val
                
            def chirp_jac(params):
                _, grad = _chirp_smooth_max(params[0], params[1], beta=50.0)
                return grad
                
            try:
                res = minimize(chirp_obj, np.array([best_alpha, best_phase]), 
                              jac=chirp_jac, method='BFGS',
                              options={'maxiter': 100, 'gtol': 1e-8})
                if res.fun < best_score:
                    h_opt = _chirp_heights(res.x[0], res.x[1])
                    _update_best(h_opt)
            except Exception:
                pass
    
    # Phase 3: Try cosine wave start and SLSQP refinement
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    def _run_slsqp(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            cs = _linear_overlap_values(x)
            m = np.max(cs)
            exp_terms = np.exp(beta * (cs - m))
            sum_exp = np.sum(exp_terms)
            return m + np.log(sum_exp) / beta
            
        try:
            res = minimize(obj_func, h0.copy(), method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    if time.time() - t0 < _DEADLINE - 20:
        _run_slsqp(h_cos, beta=50.0, maxiter=150, ftol=1e-9)
    
    # Phase 4: Final polish with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_slsqp(best_h, beta=100.0, maxiter=200, ftol=1e-12)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
