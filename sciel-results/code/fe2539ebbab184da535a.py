import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s < 1e-15:
        return np.inf
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    max_conv = np.max(conv)
    return float(max_conv / (s * s))

def smooth_max_surrogate(conv, beta=100.0):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    """
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def objective(params, beta=100.0):
    """
    Objective function for optimization.
    params: flattened array of h values (non-negative enforced via projection)
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    smoothed_peak = smooth_max_surrogate(conv, beta)
    return smoothed_peak / (s * s)

def objective_grad(params, beta=100.0):
    """
    Gradient of the objective function using FFT for efficiency.
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(params)
    
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    
    # Compute softmax weights for the max surrogate
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Gradient of smooth_max wrt h: 2 * correlate(weights, h)
    W = np.fft.fft(weights, n=n_fft)
    grad_h = 2.0 * np.real(np.fft.ifft(W * np.conj(H)))[:N]
    
    # Gradient of objective: f = smooth_max / s^2
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Zero out gradient where h is at boundary (h=0) to respect constraint
    grad_obj[h <= 1e-12] = 0.0
    
    return grad_obj

def _make_initializations():
    """Create diverse initializations for the optimization."""
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 / (2 * 0.4**2))
    gauss /= np.sum(gauss)
    inits.append(gauss)
    
    # 3. Sine bump
    sine = np.sin(np.pi * (x + 1) / 2) ** 2
    sine /= np.sum(sine)
    inits.append(sine)
    
    # 4. Triangular
    tri = 1.0 - np.abs(x)
    tri = np.maximum(tri, 0)
    tri /= np.sum(tri)
    inits.append(tri)
    
    # 5. Cosine bump (raised cosine)
    cos_bump = 0.5 * (1 + np.cos(np.pi * x))
    cos_bump /= np.sum(cos_bump)
    inits.append(cos_bump)
    
    # 6. Two-block: high-low-high
    two_block = np.zeros(N)
    two_block[:N//3] = 1.0
    two_block[2*N//3:] = 1.0
    two_block /= np.sum(two_block)
    inits.append(two_block)
    
    # 7. Linear ramp
    lin = np.linspace(0.1, 1.0, N)
    lin /= np.sum(lin)
    inits.append(lin)
    
    # 8. Random (fixed seed)
    rng = np.random.RandomState(42)
    rand_h = rng.rand(N)
    rand_h /= np.sum(rand_h)
    inits.append(rand_h)
    
    # 9. Plateau: flat center, tapered edges
    plateau = np.zeros(N)
    center = N // 2
    width = N // 4
    plateau[center-width:center+width] = 1.0
    # Smooth the edges
    for i in range(width):
        fade = i / width
        plateau[center-width+i] = fade
        plateau[center+width-1-i] = fade
    plateau /= np.sum(plateau)
    inits.append(plateau)
    
    # 10. Double Gaussian
    dg = np.exp(-((x - 0.3)**2) / (2 * 0.15**2)) + np.exp(-((x + 0.3)**2) / (2 * 0.15**2))
    dg /= np.sum(dg)
    inits.append(dg)
    
    return inits

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY_MARGIN
    
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    initializations = _make_initializations()
    bounds = [(0, None)] * N
    
    # Phase 1: Optimize each initialization with moderate beta
    for h0 in initializations:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                objective, 
                h0.copy(),
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(50.0,),
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = np.maximum(res.x, 0)
            if np.sum(h_opt) > 1e-10:
                score_opt = compute_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Refine the best solution with higher beta
    if time.time() < deadline - 20:
        try:
            res = minimize(
                objective, 
                best_h.copy(),
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(100.0,),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0)
            if np.sum(h_opt) > 1e-10:
                score_opt = compute_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with very high beta
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective, 
                best_h.copy(),
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(200.0,),
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0)
            if np.sum(h_opt) > 1e-10:
                score_opt = compute_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Coordinate descent refinement for fine-tuning
    if time.time() < deadline - 5:
        h_cd = best_h.copy()
        s = np.sum(h_cd)
        for iteration in range(100):
            if time.time() > deadline - 3:
                break
            improved = False
            # Sample a subset of indices for speed
            step = max(1, N // 200)
            for i in range(0, N, step):
                if time.time() > deadline - 3:
                    break
                old_val = h_cd[i]
                # Try small relative perturbations
                for delta_frac in [-0.1, 0.1, -0.01, 0.01]:
                    trial = h_cd.copy()
                    trial[i] = max(0, old_val * (1 + delta_frac))
                    trial_s = np.sum(trial)
                    if trial_s < 1e-10:
                        continue
                    trial_score = compute_score(trial)
                    if trial_score < best_score - 1e-12:
                        h_cd = trial
                        best_score = trial_score
                        best_h = trial.copy()
                        improved = True
                        break
                if not improved and old_val > 0:
                    # Try setting to zero
                    trial = h_cd.copy()
                    trial[i] = 0
                    if np.sum(trial) > 1e-10:
                        trial_score = compute_score(trial)
                        if trial_score < best_score - 1e-12:
                            h_cd = trial
                            best_score = trial_score
                            best_h = trial.copy()
                            improved = True
    
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
