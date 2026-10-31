import numpy as np
from scipy.fft import fft, ifft
from scipy import optimize
import time
import random

N = 600
DEADLINE = 140  # Safety margin before the 150s kill

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    if len(h) == 0:
        return 0.0
    total = np.sum(h)
    if total < 1e-10:
        return 0.0
    size = 2 * N
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*N-1])
    return np.max(conv) / (total * total)

def _smooth_max_obj_grad(h, beta):
    """
    Compute smooth max autoconvolution value and gradient w.r.t. h.
    Uses log-sum-exp with temperature 1/beta.
    Objective: f(h) = soft_max(conv(h/sum(h)))
    This approximates max(conv(h))/sum(h)^2.
    """
    h = np.maximum(h, 1e-10)
    total = np.sum(h)
    if total < 1e-10:
        return 1e10, np.zeros_like(h)
    
    # Normalize for computation of the shape-dependent part
    h_norm = h / total
    size = 2 * N
    H = fft(h_norm, size)
    conv = np.real(ifft(H * H, size)[:2*N-1])
    
    m = np.max(conv)
    shifted = (conv - m) * beta
    shifted = np.clip(shifted, -50, 50)
    exp_conv = np.exp(shifted)
    sum_exp = np.sum(exp_conv)
    weights = exp_conv / sum_exp
    
    soft_max_val = m + np.log(sum_exp) / beta
    
    # Gradient of soft_max w.r.t. h_norm
    # d(soft_max)/d(conv_k) = weights[k]
    # conv = h_norm * h_norm (linear convolution)
    # d(conv_k)/d(h_norm_j) = h_norm[k-j] + h_norm[j-k] (symmetry)
    # grad_norm[j] = sum_k weights[k] * (h_norm[k-j] + h_norm[j-k])
    # = 2 * (weights * h_norm)[j]
    
    W = fft(weights, size)
    grad_norm = np.real(ifft(W * H, size))[:N]
    grad_norm *= 2.0
    
    # Chain rule for f(h) = G(h/S)
    # df/dh_j = grad_norm[j]/S - (1/S) * sum_k grad_norm[k] * h_norm[k]
    sum_grad_u = np.dot(grad_norm, h_norm)
    grad_h = (grad_norm - sum_grad_u) / total
    
    return soft_max_val, grad_h

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Initialize with uniform
    best_h = np.ones(N) / N
    best_val = compute_autoconv_max(best_h)
    
    # Stage 1: Parametric Global Search to find a good initial shape
    # We use a simple Gaussian Mixture Model (GMM) parameterization for global search
    # because it is smooth and allows efficient DE optimization.
    K = 8  # Number of Gaussian components
    
    i = np.arange(N, dtype=float)
    
    def gmm_to_h(params):
        """Convert GMM parameters to height array."""
        h = np.zeros(N)
        for j in range(K):
            w_j = params[3*j]
            c_j = params[3*j+1]
            s_j = max(params[3*j+2], 1.0)
            
            # Gaussian bump
            bump = w_j * np.exp(-((i - c_j)**2) / (2 * s_j**2))
            h += bump
        return np.maximum(h, 0)
    
    def gmm_objective(params):
        if time_left() < 2:
            return 1e10
        h = gmm_to_h(params)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h_norm = h / s
        return compute_autoconv_max(h_norm)
    
    # Define bounds for GMM parameters
    bounds = []
    for j in range(K):
        bounds.append((0.1, 5.0))    # weight
        bounds.append((0.0, N-1))    # center
        bounds.append((5.0, N/3))    # width (sigma)
    
    # Run Differential Evolution
    if time_left() > 60:
        pop_size = min(30, max(20, int(time_left() * 0.25)))
        max_iter = int(min(60, time_left() * 0.5))
        
        try:
            # Use latin hypercube for better initial spread
            result = optimize.differential_evolution(
                gmm_objective,
                bounds=bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-8,
                seed=42,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            
            if result.fun < best_val:
                h_de = gmm_to_h(result.x)
                s = np.sum(h_de)
                if s > 1e-10:
                    h_de = h_de / s
                    val_de = compute_autoconv_max(h_de)
                    if val_de < best_val:
                        best_val = val_de
                        best_h = h_de.copy()
        except Exception:
            pass
    
    # Stage 2: High-precision Local Refinement using Annealed L-BFGS-B
    # This is the structural change: optimizing the full 600-dim vector
    # using the analytic gradient of the smooth max surrogate.
    if time_left() > 20:
        current_h = best_h.copy()
        current_val = best_val
        
        # Annealing schedule for beta (sharpening the max approximation)
        betas = [10.0, 50.0, 200.0, 500.0, 1000.0, 2000.0]
        
        for beta in betas:
            if time_left() < 5:
                break
            
            try:
                # Define objective and gradient for L-BFGS-B
                def obj(h):
                    return _smooth_max_obj_grad(h, beta)[0]
                
                def jac(h):
                    return _smooth_max_obj_grad(h, beta)[1]
                
                res = optimize.minimize(
                    obj,
                    current_h,
                    jac=jac,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
                )
                
                # Extract result
                new_h = np.maximum(res.x, 0)
                s = np.sum(new_h)
                if s < 1e-10:
                    continue
                new_h = new_h / s
                new_val = compute_autoconv_max(new_h)
                
                if new_val < current_val - 1e-10:
                    current_val = new_val
                    current_h = new_h.copy()
            except Exception:
                pass
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            
    # Stage 3: Final Polish with very high beta for a few iterations
    if time_left() > 5:
        try:
            beta_final = 5000.0
            def obj_final(h):
                return _smooth_max_obj_grad(h, beta_final)[0]
            def jac_final(h):
                return _smooth_max_obj_grad(h, beta_final)[1]
                
            res = optimize.minimize(
                obj_final,
                best_h,
                jac=jac_final,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 200}
            )
            new_h = np.maximum(res.x, 0)
            s = np.sum(new_h)
            if s > 1e-10:
                new_h = new_h / s
                new_val = compute_autoconv_max(new_h)
                if new_val < best_val:
                    best_val = new_val
                    best_h = new_h.copy()
        except Exception:
            pass

    return list(best_h)
