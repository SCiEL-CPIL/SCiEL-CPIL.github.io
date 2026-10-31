import numpy as np
import time
from scipy import optimize
from scipy.fft import fft, ifft

N = 600

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*n-1])
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total < 1e-10:
        return 0.0
    conv = _fft_convolve(h)
    return np.max(conv) / (total * total)

def _smooth_max_obj_grad(h, beta):
    """
    Compute smooth max autoconvolution value and gradient w.r.t. h.
    Uses log-sum-exp with temperature 1/beta.
    """
    h = np.maximum(h, 1e-10)
    total = np.sum(h)
    if total < 1e-10:
        return 1e10, np.zeros_like(h)
    
    # Normalize for computation of the shape-dependent part
    h_norm = h / total
    conv = _fft_convolve(h_norm)
    
    m = np.max(conv)
    shifted = (conv - m) * beta
    shifted = np.clip(shifted, -50, 50)
    exp_conv = np.exp(shifted)
    sum_exp = np.sum(exp_conv)
    weights = exp_conv / sum_exp
    
    soft_max_val = m + np.log(sum_exp) / beta
    
    # Gradient of soft_max w.r.t. h_norm
    # grad[j] = 2 * (weights * h_norm)[j] (linear convolution)
    size = 2 * N
    W = fft(weights, size)
    H = fft(h_norm, size)
    grad_norm = np.real(ifft(W * H, size))[:N]
    grad_norm *= 2.0
    
    # Gradient chain rule for f(h) = soft_max(conv(h/S))
    sum_grad_u = np.dot(grad_norm, h_norm)
    grad_h = (grad_norm - sum_grad_u) / total
    
    return soft_max_val, grad_h

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def time_left():
        return deadline - time.time()

    # Initial guess: all ones (this gives score = 1.0)
    best_h = np.ones(N) / N
    best_score = compute_autoconv_max(best_h)
    
    # Strategy 1: Phase-based representation + Differential Evolution
    # Represent h as |z|^2 where z = sum_k c_k * exp(i * phase_k * t)
    # This ensures h is non-negative and has spectral structure
    
    def phase_to_h(phases, amps):
        """Convert phases and amplitudes to non-negative step function."""
        t = np.linspace(0, 1, N)
        z = np.zeros(N, dtype=complex)
        for k in range(len(phases)):
            z += amps[k] * np.exp(1j * phases[k] * t)
        h = np.abs(z)**2
        h = np.maximum(h, 0)
        total = np.sum(h)
        if total > 0:
            h /= total
        else:
            h = np.ones(N) / N
        return h

    def phase_objective(params):
        """Objective for phase optimization."""
        n_phases = len(params) // 2
        phases = params[:n_phases]
        amps = params[n_phases:]
        h = phase_to_h(phases, amps)
        return compute_autoconv_max(h)

    # Try a few different numbers of phase components
    best_phase_h = None
    best_phase_score = best_score
    
    for n_phases in [2, 3, 4]:
        if time_left() < 30:
            break
        
        # Bounds: phases in [0, 2*pi), amps in [0.1, 2.0]
        bounds = [(0, 2*np.pi)] * n_phases + [(0.1, 2.0)] * n_phases
        
        try:
            max_iter_de = int(min(40, time_left() // 2.0))
            if max_iter_de < 10:
                max_iter_de = 10
                
            de_result = optimize.differential_evolution(
                phase_objective,
                bounds=bounds,
                seed=42 + n_phases,
                maxiter=max_iter_de,
                popsize=12,
                tol=1e-5,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            
            if de_result.fun < best_phase_score:
                phases = de_result.x[:n_phases]
                amps = de_result.x[n_phases:]
                h_phase = phase_to_h(phases, amps)
                best_phase_score = de_result.fun
                best_phase_h = h_phase
        except Exception:
            pass

    if best_phase_h is not None and best_phase_score < best_score:
        best_score = best_phase_score
        best_h = best_phase_h.copy()

    # Strategy 2: GMM Initialization + DE (as before)
    K = 4  # Number of Gaussians
    
    def gmm_to_h(params):
        """Convert GMM parameters to discrete heights."""
        h = np.zeros(N)
        i = np.arange(N)
        for k in range(K):
            w_k = max(params[3 * k], 0.01)
            mu_k = params[3 * k + 1]
            sig_k = max(params[3 * k + 2], 1.0)
            h += w_k * np.exp(-((i - mu_k) ** 2) / (2 * sig_k ** 2))
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            h /= np.sum(h)
        else:
            h = np.ones(N) / N
        return h

    def gmm_objective(params):
        h = gmm_to_h(params)
        return compute_autoconv_max(h)

    bounds_gmm = []
    for k in range(K):
        bounds_gmm.append((0.1, 5.0))   # weight
        bounds_gmm.append((0, N - 1))   # mean
        bounds_gmm.append((5.0, N / 3.0))  # sigma

    if time_left() > 40:
        try:
            max_iter_de = int(min(30, time_left() // 2.0))
            if max_iter_de < 10:
                max_iter_de = 10
                
            de_result = optimize.differential_evolution(
                gmm_objective,
                bounds=bounds_gmm,
                seed=123,
                maxiter=max_iter_de,
                popsize=12,
                tol=1e-5,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            if de_result.fun < best_score:
                h_de = gmm_to_h(de_result.x)
                best_score = de_result.fun
                best_h = h_de.copy()
        except Exception:
            pass

    # Strategy 3: Annealed L-BFGS-B Polish
    if time_left() > 20 and best_h is not None:
        betas = [10.0, 50.0, 200.0, 500.0, 1000.0]
        
        current_h = best_h.copy()
        current_val = best_score
        
        for beta in betas:
            if time_left() < 4:
                break
            
            try:
                res = optimize.minimize(
                    lambda h: _smooth_max_obj_grad(h, beta)[0],
                    current_h,
                    jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 150}
                )
                
                if res.success:
                    final_h = np.maximum(res.x, 0)
                    if np.sum(final_h) > 0:
                        final_h /= np.sum(final_h)
                        final_val = compute_autoconv_max(final_h)
                        if final_val < current_val:
                            current_val = final_val
                            current_h = final_h.copy()
            except Exception:
                pass
        
        if current_val < best_score:
            best_score = current_val
            best_h = current_h.copy()

    # Strategy 4: Multi-restart with random phase perturbations
    if time_left() > 15:
        n_restarts = int(min(3, time_left() // 5))
        for r in range(n_restarts):
            if time_left() < 6:
                break
            
            # Generate random phase parameters
            n_phases = 3
            params = np.concatenate([
                np.random.uniform(0, 2*np.pi, n_phases),
                np.random.uniform(0.1, 2.0, n_phases)
            ])
            
            h_init = phase_to_h(params[:n_phases], params[n_phases:])
            
            try:
                current_h = h_init
                current_val = compute_autoconv_max(current_h)
                
                for beta in [50.0, 200.0]:
                    if time_left() < 3:
                        break
                    res = optimize.minimize(
                        lambda h: _smooth_max_obj_grad(h, beta)[0],
                        current_h,
                        jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 80}
                    )
                    
                    if res.success:
                        final_h = np.maximum(res.x, 0)
                        if np.sum(final_h) > 0:
                            final_h /= np.sum(final_h)
                            final_val = compute_autoconv_max(final_h)
                            if final_val < current_val:
                                current_val = final_val
                                current_h = final_h.copy()
                
                if current_val < best_score:
                    best_score = current_val
                    best_h = current_h.copy()
            except Exception:
                pass

    return list(best_h)
