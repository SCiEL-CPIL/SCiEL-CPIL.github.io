"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac


def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Due to symmetry, max |ac[1:N]| is sufficient
    return np.max(np.abs(ac[1:N]))


def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    if max_val < 1e-10:
        return max_val
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta


def _tanh_relaxation_objective(z, beta=50.0):
    """
    Objective for tanh relaxation: minimize soft_max_sidelobe(tanh(z))
    Returns (value, gradient_wrt_z)
    """
    h = np.tanh(z)
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient of soft_max w.r.t. h
    weights = beta * exp_shifted / sum_exp
    grad_h = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        if c_k == 0:
            continue
        grad_h[0:N-k] += c_k * h[k:N]
        grad_h[k:N] += c_k * h[0:N-k]
    
    # Chain rule: d/dz = d/dh * dh/dz = grad_h * (1 - tanh(z)^2)
    sech2 = 1.0 - h**2
    grad_z = grad_h * sech2
    
    return obj, grad_z


def _spectral_shaping_objective(phi, beta=30.0):
    """
    Optimize the phase of the power spectrum to minimize sidelobe peaks.
    
    We parameterize the power spectrum as |F(k)|^2 = base_power * exp(phi(k))
    where phi is a real-valued vector we optimize. The constraint is that
    the inverse FFT gives a valid signal (we'll handle the mapping to ±1 later).
    
    For simplicity, we optimize a smooth surrogate: minimize the soft-max of
    the sidelobes of the autocorrelation computed from the shaped spectrum.
    
    Returns (objective_value, gradient_wrt_phi)
    """
    # phi has length N//2+1 for rfft
    n_fft = 2 * N
    n_freq = N + 1  # rfft length
    
    # Start from a flat spectrum and add the shape
    # We need to ensure the resulting signal has zero mean or some constraint
    # For now, let's just shape the spectrum and see what we get
    
    # The power spectrum in the frequency domain
    # We want to minimize the peak of the inverse FFT of |F|^2
    
    # Let F(k) = sqrt(base_power) * exp(phi(k)/2) * exp(i*theta(k))
    # For simplicity, assume theta=0 and optimize phi
    
    # Actually, let's use a different approach: optimize the actual signal h
    # in a continuous relaxation, then threshold
    
    # This function is a placeholder for now
    return 0.0, np.zeros(n_freq)


def _continuous_spectral_optimize(beta=30.0, max_iter=200, time_limit=10.0):
    """
    Optimize a continuous signal h in [-1, 1] to minimize sidelobe peaks,
    using spectral shaping as a guide.
    
    Returns the continuous signal h.
    """
    start_time = time.time()
    
    # Initialize with a random continuous signal
    rng = np.random.default_rng(42)
    h = rng.uniform(-1, 1, size=N)
    
    # Use L-BFGS-B to minimize the soft-max sidelobe objective
    def obj_and_grad(h):
        ac = _compute_autocorr(h)
        sidelobes_abs = np.abs(ac[1:N])
        
        max_val = np.max(sidelobes_abs)
        if max_val < 1e-10:
            return max_val, np.zeros(N)
        
        shifted = beta * (sidelobes_abs - max_val)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        obj = max_val + np.log(sum_exp) / beta
        
        # Gradient w.r.t. h
        weights = beta * exp_shifted / sum_exp
        grad_h = np.zeros(N)
        
        for k in range(1, N):
            w_k = weights[k-1]
            if w_k < 1e-15:
                continue
            sign_k = np.sign(ac[k])
            c_k = sign_k * w_k
            if c_k == 0:
                continue
            grad_h[0:N-k] += c_k * h[k:N]
            grad_h[k:N] += c_k * h[0:N-k]
        
        return obj, grad_h
    
    try:
        res = minimize(obj_and_grad, h, method='L-BFGS-B',
                       bounds=[(-1.0, 1.0)] * N,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        h_opt = res.x
    except Exception:
        h_opt = h
    
    return h_opt


def _threshold_continuous(h_cont, threshold=0.0):
    """
    Map a continuous signal to a signed step function via thresholding.
    """
    h_bin = np.sign(h_cont - threshold)
    zero_mask = (h_bin == 0)
    if np.any(zero_mask):
        # For zeros, use a deterministic rule based on position
        h_bin[zero_mask] = np.where(np.arange(N)[zero_mask] % 2 == 0, 1.0, -1.0)
    return h_bin


def _multi_flip_local_search(h, max_iterations=5000, time_limit=5.0, n_flips=1):
    """
    Local search that tries flipping n_flips bits at a time.
    Uses FFT for evaluation.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping n_flips random bits
        indices = np.random.choice(N, size=n_flips, replace=False)
        old_vals = h[indices].copy()
        h[indices] = -h[indices]
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            h[indices] = old_vals
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous spectral optimization with multiple restarts
    num_spectral_starts = 6
    betas_spectral = [20.0, 30.0, 50.0]
    
    for i in range(num_spectral_starts):
        if time.time() >= deadline - 50.0:
            break
        
        beta = betas_spectral[i % len(betas_spectral)]
        
        # Continuous optimization
        h_cont = _continuous_spectral_optimize(beta=beta, max_iter=150, time_limit=8.0)
        
        # Threshold to binary
        h_bin = _threshold_continuous(h_cont)
        
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
        
        # Quick 1-bit local search
        if time.time() < deadline - 30.0:
            h_ref, val_ref = _multi_flip_local_search(h_bin.copy(), max_iterations=3000, time_limit=3.0, n_flips=1)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Phase 2: Tanh Relaxation with L-BFGS-B (original approach)
    num_starts = 6
    betas = [20.0, 50.0, 100.0]
    
    for i in range(num_starts):
        if time.time() >= deadline - 30.0:
            break
        
        rng = np.random.default_rng(1000 + i)
        # Initialize z so that tanh(z) is near ±1 but not exactly
        # This allows gradient flow
        z0 = rng.normal(0, 1, size=N) * 2.0
        
        beta = betas[i % len(betas)]
        
        def obj_grad(z, beta=beta):
            return _tanh_relaxation_objective(z, beta)
        
        try:
            res = minimize(obj_grad, z0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30})
        except Exception:
            continue
        
        # Convert back to binary
        h_bin = np.sign(np.tanh(res.x))
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = rng.choice([-1.0, 1.0], size=np.sum(zero_mask))
        
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
        
        # Quick 1-bit local search
        if time.time() < deadline - 20.0:
            h_ref, val_ref = _multi_flip_local_search(h_bin.copy(), max_iterations=2000, time_limit=2.0, n_flips=1)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Phase 3: 2-bit flip local search for better escape
    if best_h is not None and time.time() < deadline - 15.0:
        h_ref, val_ref = _multi_flip_local_search(best_h.copy(), max_iterations=5000, time_limit=8.0, n_flips=2)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 4: 3-bit flip local search
    if best_h is not None and time.time() < deadline - 8.0:
        h_ref, val_ref = _multi_flip_local_search(best_h.copy(), max_iterations=3000, time_limit=5.0, n_flips=3)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 5: Final 1-bit polish
    if best_h is not None and time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 1.0:
            h_ref, val_ref = _multi_flip_local_search(best_h.copy(), max_iterations=10000, time_limit=remaining, n_flips=1)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
