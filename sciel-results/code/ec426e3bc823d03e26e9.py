import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Use FFT for efficient convolution
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    Returns: (1/beta) * log(sum(exp(beta * conv)))
    """
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

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
    
    # Compute gradient using FFT
    # grad_h = 2 * correlate(weights, h)
    W = np.fft.fft(weights, n=n_fft)
    
    # Correlation via FFT
    grad_h = 2.0 * np.real(np.fft.ifft(W * np.conj(H)))[:N]
    
    # Gradient of objective: f = smooth_max / s^2
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Zero out gradient where h is at boundary (h=0) to respect constraint
    grad_obj[h <= 1e-12] = 0.0
    
    return grad_obj

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    # Parametric search for "top-hat with tapered edges"
    # h(x) = max(0, 1 - (|x|/W)^(2n))^(1/(2n))
    # x is in [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, N, endpoint=False) # Using endpoint=False to match interval logic? 
    # The problem says "600 equal intervals on [-1/4, 1/4]".
    # Usually this means bins. Let's assume centers or just uniform grid.
    # The current code uses np.linspace(-1, 1, N) in init, but compute_score just takes the array.
    # The geometry is fixed by the evaluator. We just return heights.
    # For parametric generation, we need to map parameters to the 600 bins.
    # Let's assume the bins are centered at x_i = -0.25 + (i+0.5)*dx, dx=0.5/600.
    dx = 0.5 / N
    x_centers = -0.25 + (np.arange(N) + 0.5) * dx

    def gen_parametric(params):
        W, n = params
        if W <= 0 or n <= 0:
            return np.ones(N) / N
        # Super-ellipse window
        # 1 - (|x|/W)^(2n)
        # If |x| > W, value is negative -> 0
        # We clip the base to 0
        base = 1.0 - np.abs(x_centers / W) ** (2 * n)
        base = np.maximum(base, 0.0)
        h = base ** (1.0 / (2 * n))
        # Normalize? The objective is invariant to scaling if we consider max_conv / sum^2.
        # max(a*h * a*h) / (sum a*h)^2 = a^2 max_conv / a^2 sum^2 = max_conv/sum^2.
        # So normalization doesn't matter for the score, but helps numerical stability.
        s = np.sum(h)
        if s > 0:
            h /= s
        else:
            h = np.ones(N) / N
        return h

    def parametric_objective(params):
        h = gen_parametric(params)
        return compute_score(h)

    # Search range for W and n
    # W is width. Max width is 0.5 (full support).
    # n is exponent. n=0.5 is triangle-like, n=1 is flat-top-like (if modified), n->inf is box.
    # Let's search W in [0.05, 0.5] and n in [0.2, 5.0]
    
    try:
        if time.time() < deadline - 30:
            # Differential Evolution for global search in 2D
            bounds = [(0.05, 0.5), (0.2, 5.0)]
            res_de = differential_evolution(
                parametric_objective, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                seed=42,
                tol=1e-8,
                polish=True
            )
            best_param = res_de.x
            h_param = gen_parametric(best_param)
            score_param = compute_score(h_param)
            
            if score_param < best_score:
                best_score = score_param
                best_h = h_param
    except Exception:
        pass

    # Refine the parametric solution with local optimizer (Nelder-Mead)
    try:
        if time.time() < deadline - 20:
            # Start from best parametric or uniform
            if 'best_param' in locals():
                x0 = best_param
            else:
                x0 = [0.25, 1.0]
            
            res_nm = minimize(
                parametric_objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            h_param = gen_parametric(res_nm.x)
            score_param = compute_score(h_param)
            
            if score_param < best_score:
                best_score = score_param
                best_h = h_param
    except Exception:
        pass

    # Now use the best_h as initialization for full 600-variable optimization
    np.random.seed(42)
    
    # Multiple initializations, including the parametric best
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N) / N)
    
    # 2. Best Parametric
    initializations.append(best_h.copy())
    
    # 3. Gaussian
    x_gauss = np.linspace(-1, 1, N)
    gauss = np.exp(-x_gauss**2 / (2 * 0.5**2))
    gauss /= np.sum(gauss)
    initializations.append(gauss)
    
    # 4. Sine
    sine = np.sin(np.pi * (x_gauss + 1) / 2) ** 2
    sine /= np.sum(sine)
    initializations.append(sine)
    
    # 5. Random
    rand_h = np.random.rand(N)
    rand_h /= np.sum(rand_h)
    initializations.append(rand_h)
    
    bounds = [(0, None)] * N
    
    for h0 in initializations:
        if time.time() > deadline - 25:
            break
            
        # Optimize with adaptive beta
        try:
            # Start with lower beta for smoother landscape
            res = minimize(
                objective, 
                h0, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(50.0,),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            
            # Refine with higher beta
            if time.time() < deadline - 20:
                res2 = minimize(
                    objective, 
                    h_opt, 
                    method='L-BFGS-B', 
                    jac=objective_grad, 
                    args=(100.0,),
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = res2.x
                h_opt = np.maximum(h_opt, 0)
            
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Final polish with tighter tolerance and higher beta
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(200.0,),
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Coordinate descent refinement with adaptive step size
    if time.time() < deadline - 5:
        h_cd = best_h.copy()
        step_size = 0.005  # Start with smaller steps
        
        for iteration in range(200):
            if time.time() > deadline - 3:
                break
            improved = False
            
            # Focus on indices near the peak of autoconvolution
            n_fft = 2 * N - 1
            H = np.fft.fft(h_cd, n=n_fft)
            conv = np.real(np.fft.ifft(H * H))[:2*N-1]
            peak_idx = np.argmax(conv)
            
            # Priority: indices that contribute most to the peak
            # The peak at index k gets contributions from h[i]*h[k-i] for valid i
            priority_order = list(range(N))
            
            for i in priority_order:
                if time.time() > deadline - 3:
                    break
                
                old_val = h_cd[i]
                trial = h_cd.copy()
                
                # Try increasing
                trial[i] = old_val + step_size
                trial /= np.sum(trial)
                trial_score = compute_score(trial)
                if trial_score < best_score - 1e-12:
                    h_cd = trial
                    best_score = trial_score
                    best_h = trial.copy()
                    improved = True
                    continue
                
                # Try decreasing
                trial = h_cd.copy()
                trial[i] = max(0, old_val - step_size)
                trial /= np.sum(trial)
                trial_score = compute_score(trial)
                if trial_score < best_score - 1e-12:
                    h_cd = trial
                    best_score = trial_score
                    best_h = trial.copy()
                    improved = True
                    continue
                
                # Try setting to zero if currently positive
                if old_val > 1e-6:
                    trial = h_cd.copy()
                    trial[i] = 0
                    if np.sum(trial) > 0:
                        trial /= np.sum(trial)
                        trial_score = compute_score(trial)
                        if trial_score < best_score - 1e-12:
                            h_cd = trial
                            best_score = trial_score
                            best_h = trial.copy()
                            improved = True
                            continue
            
            # Adapt step size
            if improved:
                step_size = max(0.001, step_size * 0.7)
            else:
                step_size = min(0.02, step_size * 1.5)
            
            if not improved and step_size < 0.001:
                break

    # Final high-precision polish with very high beta
    if time.time() < deadline - 2:
        try:
            res = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(500.0,),
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
# EVOLVE-BLOCK-END
