"""Combined program: chirp + Zadoff-Chu + non-harmonic sinusoid basis + smooth-max L-BFGS-B refinement.

Strategy:
- Keep the chirp-based parameterization as global search space.
- Add Zadoff-Chu inspired sequences as additional candidates.
- Use differential evolution for global search over chirp parameters.
- Refine using L-BFGS-B on a non-harmonic sinusoid basis with arbitrary frequencies.
- Also try Nelder-Mead polish on chirp parameters.
- Deadline guard ensures we return before 150s.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s wall-clock

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _smooth_max_value(h, alpha=50.0):
    """Smooth surrogate for the max of convolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    # Soft max via log-sum-exp
    scaled = conv * alpha
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / alpha
    return abs(2.0 * N * soft_max / (s * s))

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    # Quadratic phase: phase + rate * (i/N)^2
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _zadoff_chu_h(params):
    """Generate h from Zadoff-Chu-like parameters with windowing: [phase, rate, amp, base, win_center, win_width, win_type].
    
    Zadoff-Chu sequences have constant amplitude and near-zero non-zero
    autocorrelation. We use h[i] = base + amp * window(i) * cos(phase + rate * i * (i - N/2) / N)
    which approximates the quadratic phase of ZC sequences with a smooth window to suppress sidelobes.
    """
    phase, rate, amp, base, win_center, win_width, win_type = params
    i = np.arange(N, dtype=float)
    # ZC-like quadratic phase centered
    ang = phase + rate * i * (i - N / 2.0) / N
    # Apply window function
    if win_type < 0.5:
        # Hann window
        w = 0.5 * (1 + np.cos(2 * np.pi * (i - win_center) / (win_width * N)))
    else:
        # Gaussian window
        w = np.exp(-0.5 * ((i - win_center) / (win_width * N)) ** 2)
    h = base + amp * w * np.cos(ang)
    return h

def _zadoff_chu_obj(params):
    """Objective for Zadoff-Chu parameters."""
    h = _zadoff_chu_h(params)
    return _value(h)

def _zadoff_chu_smooth_obj(params, alpha=50.0):
    """Smooth objective for Zadoff-Chu parameters."""
    h = _zadoff_chu_h(params)
    return _smooth_max_value(h, alpha)

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

# Non-harmonic sinusoid basis for refinement
# h[i] = base + sum_j a_j * cos(2*pi*f_j*i + phi_j)
# Parameters: [f_1, phi_1, a_1, f_2, phi_2, a_2, ..., base]
_K_NH = 8  # number of non-harmonic sinusoids

def _nh_h(params):
    """Generate h from non-harmonic sinusoid parameters.
    params: [f_1, phi_1, a_1, f_2, phi_2, a_2, ..., base]
    """
    i = np.arange(N, dtype=float)
    base_val = params[-1]
    h = np.full(N, base_val)
    for j in range(_K_NH):
        f_j = params[3*j]
        phi_j = params[3*j + 1]
        a_j = params[3*j + 2]
        h += a_j * np.cos(2.0 * np.pi * f_j * i + phi_j)
    return h

def _nh_obj(params):
    """Objective for non-harmonic sinusoid parameters."""
    h = _nh_h(params)
    return _value(h)

def _nh_smooth_obj(params, alpha=50.0):
    """Smooth objective for non-harmonic sinusoid parameters."""
    h = _nh_h(params)
    return _smooth_max_value(h, alpha)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # Phase 1: Global search via differential evolution on chirp parameters
    # params: [phase, rate, amp, base]
    bounds = [
        (-np.pi, np.pi),      # phase
        (-200.0, 200.0),      # rate (quadratic phase coefficient)
        (-3.0, 3.0),          # amplitude
        (-3.0, 3.0),          # base
    ]
    
    try:
        if deadline_ok():
            res = differential_evolution(
                _chirp_obj,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=10,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res.x is not None:
                h = _chirp_h(res.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
        pass
    
    # Phase 1b: Global search via differential evolution on Zadoff-Chu parameters with windowing
    if deadline_ok():
        try:
            bounds_zc = [
                (-np.pi, np.pi),      # phase
                (-50.0, 50.0),        # rate (ZC quadratic coefficient)
                (-3.0, 3.0),          # amplitude
                (-3.0, 3.0),          # base
                (0.0, 1.0),           # window center (fraction of N)
                (0.2, 2.0),           # window width (fraction of N)
                (0.0, 1.0),           # window type (0=Hann, 1=Gaussian)
            ]
            res_zc = differential_evolution(
                _zadoff_chu_obj,
                bounds_zc,
                seed=123,
                maxiter=40,
                tol=1e-6,
                popsize=10,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res_zc.x is not None:
                h = _zadoff_chu_h(res_zc.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 1c: Nelder-Mead polish on Zadoff-Chu parameters
    if deadline_ok():
        try:
            for start in [
                [0.0, 10.0, 1.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 20.0, 1.0, 0.0, 0.5, 0.3, 0.0],
                [np.pi/4, 15.0, 1.5, 0.0, 0.5, 0.7, 1.0],
                [0.0, 30.0, 1.0, 0.0, 0.5, 0.4, 1.0],
            ]:
                if not deadline_ok():
                    break
                res_zc_nm = minimize(
                    _zadoff_chu_obj,
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200, 'maxfev': 300},
                )
                if res_zc_nm.x is not None:
                    h = _zadoff_chu_h(res_zc_nm.x)
                    v = _value(h)
                    if v < best_v:
                        best_v = v
                        best_h = h.copy()
        except Exception:
            pass
    
    # Phase 1d: L-BFGS-B refinement on Zadoff-Chu parameters with smooth max
    if deadline_ok():
        try:
            # Start from a reasonable default
            x0_zc = [0.0, 10.0, 1.0, 0.0, 0.5, 0.5, 0.0]
            
            res_zc_lbfgs = minimize(
                _zadoff_chu_smooth_obj,
                x0_zc,
                method='L-BFGS-B',
                bounds=bounds_zc,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res_zc_lbfgs.x is not None:
                h = _zadoff_chu_h(res_zc_lbfgs.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish on chirp parameters
    if deadline_ok():
        try:
            # Start from the best found, or from a reasonable default
            start_params = [0.0, 10.0, 1.0, 1.0]
            # If we have a better best_h, try to estimate chirp params from it
            # For simplicity, just run Nelder-Mead from a few starting points
            for start in [
                [0.0, 10.0, 1.0, 1.0],
                [0.0, -10.0, 1.0, 1.0],
                [np.pi/2, 50.0, 0.5, 1.0],
                [0.0, 100.0, 1.5, 0.5],
            ]:
                if not deadline_ok():
                    break
                res2 = minimize(
                    _chirp_obj,
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200, 'maxfev': 300},
                )
                if res2.x is not None:
                    h = _chirp_h(res2.x)
                    v = _value(h)
                    if v < best_v:
                        best_v = v
                        best_h = h.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B refinement on non-harmonic sinusoid basis with smooth max surrogate
    if deadline_ok():
        try:
            # Initialize non-harmonic parameters from a few reasonable starting points
            # params: [f_1, phi_1, a_1, f_2, phi_2, a_2, ..., base]
            # Try a few different frequency sets
            freq_sets = [
                [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # half-integer frequencies
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # integer frequencies
                [0.3, 0.7, 1.2, 1.8, 2.3, 2.9, 3.4, 4.1],  # irregular
            ]
            
            for freq_set in freq_sets:
                if not deadline_ok():
                    break
                x0 = np.zeros(3 * _K_NH + 1)
                for j in range(_K_NH):
                    x0[3*j] = freq_set[j]
                    x0[3*j + 1] = 0.0  # phase
                    x0[3*j + 2] = 0.5  # amplitude
                x0[-1] = 0.0  # base
                
                res3 = minimize(
                    _nh_smooth_obj,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                if res3.x is not None:
                    h = _nh_h(res3.x)
                    v = _value(h)
                    if v < best_v:
                        best_v = v
                        best_h = h.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish on non-harmonic sinusoid basis
    if deadline_ok():
        try:
            # Re-initialize from current best_h by least-squares fitting
            h_best = best_h
            mean_h = np.mean(h_best)
            i = np.arange(N, dtype=float)
            
            # Build design matrix for non-harmonic basis
            # Use a moderate number of trial frequencies
            trial_freqs = np.linspace(0.1, 10.0, 20)
            A = np.zeros((N, 3 * _K_NH + 1))
            for j in range(_K_NH):
                f_j = trial_freqs[j]
                A[:, 3*j] = np.cos(2.0 * np.pi * f_j * i)
                A[:, 3*j + 1] = -2.0 * np.pi * f_j * i * np.sin(2.0 * np.pi * f_j * i)  # derivative w.r.t. phase
                A[:, 3*j + 2] = np.cos(2.0 * np.pi * f_j * i)  # derivative w.r.t. amplitude
            A[:, -1] = 1.0  # base
            
            # Actually, let's just use a simpler approach: fit with fixed frequencies
            # and optimize amplitudes, phases, and base
            fixed_freqs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            A = np.zeros((N, 2 * _K_NH + 1))
            for j in range(_K_NH):
                f_j = fixed_freqs[j]
                A[:, 2*j] = np.cos(2.0 * np.pi * f_j * i)
                A[:, 2*j + 1] = np.sin(2.0 * np.pi * f_j * i)
            A[:, -1] = 1.0
            
            try:
                sol = np.linalg.lstsq(A, h_best, rcond=None)[0]
                # Convert to non-harmonic parameter format
                x0 = np.zeros(3 * _K_NH + 1)
                for j in range(_K_NH):
                    c_j = sol[2*j]
                    s_j = sol[2*j + 1]
                    a_j = np.sqrt(c_j**2 + s_j**2)
                    phi_j = np.arctan2(-s_j, c_j)  # cos(x + phi) = cos(x)cos(phi) - sin(x)sin(phi)
                    x0[3*j] = fixed_freqs[j]
                    x0[3*j + 1] = phi_j
                    x0[3*j + 2] = a_j
                x0[-1] = sol[-1]
            except:
                x0 = np.zeros(3 * _K_NH + 1)
                for j in range(_K_NH):
                    x0[3*j] = fixed_freqs[j]
                    x0[3*j + 1] = 0.0
                    x0[3*j + 2] = 0.5
                x0[-1] = 0.0
            
            res4 = minimize(
                _nh_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 1000},
            )
            if res4.x is not None:
                h = _nh_h(res4.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish using smooth max with tighter alpha
    if deadline_ok():
        try:
            # Use the best non-harmonic parameters found so far
            # Re-fit from current best_h
            h_best = best_h
            mean_h = np.mean(h_best)
            i = np.arange(N, dtype=float)
            
            fixed_freqs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            A = np.zeros((N, 2 * _K_NH + 1))
            for j in range(_K_NH):
                f_j = fixed_freqs[j]
                A[:, 2*j] = np.cos(2.0 * np.pi * f_j * i)
                A[:, 2*j + 1] = np.sin(2.0 * np.pi * f_j * i)
            A[:, -1] = 1.0
            
            try:
                sol = np.linalg.lstsq(A, h_best, rcond=None)[0]
                x0 = np.zeros(3 * _K_NH + 1)
                for j in range(_K_NH):
                    c_j = sol[2*j]
                    s_j = sol[2*j + 1]
                    a_j = np.sqrt(c_j**2 + s_j**2)
                    phi_j = np.arctan2(-s_j, c_j)
                    x0[3*j] = fixed_freqs[j]
                    x0[3*j + 1] = phi_j
                    x0[3*j + 2] = a_j
                x0[-1] = sol[-1]
            except:
                x0 = np.zeros(3 * _K_NH + 1)
                for j in range(_K_NH):
                    x0[3*j] = fixed_freqs[j]
                    x0[3*j + 1] = 0.0
                    x0[3*j + 2] = 0.5
                x0[-1] = 0.0
            
            res5 = minimize(
                lambda p: _nh_smooth_obj(p, alpha=200.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res5.x is not None:
                h = _nh_h(res5.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
