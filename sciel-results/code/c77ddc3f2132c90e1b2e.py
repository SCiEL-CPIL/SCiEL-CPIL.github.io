import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def density_from_params(params, N=N):
    """
    Generate a nonnegative density from continuous parameters.
    params: (mu, sigma, peak, width) for a mixture of two Gaussians
    or (mu1, sigma1, mu2, sigma2, w1, w2) for a two-component mixture.
    """
    # Use a two-component Gaussian mixture
    mu1, sigma1, mu2, sigma2, w1, w2 = params
    x = np.linspace(-0.25, 0.25, N)
    
    # Ensure positive sigmas
    sigma1 = max(abs(sigma1), 0.01)
    sigma2 = max(abs(sigma2), 0.01)
    
    # Gaussian components
    g1 = np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    g2 = np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    
    # Weights (softplus for positivity)
    w1 = np.exp(w1)
    w2 = np.exp(w2)
    
    h = w1 * g1 + w2 * g2
    
    # Ensure nonnegativity and normalize
    h = np.maximum(h, 0.0)
    total = np.sum(h)
    if total < 1e-15:
        h = np.zeros(N)
        h[0] = 1.0
    else:
        h = h / total
    
    return h

def continuous_objective(params, N=N):
    """
    Objective for continuous optimization: minimize -log(ratio).
    """
    h = density_from_params(params, N)
    ratio = compute_ratio(h)
    if ratio < 1e-15:
        return 1e10, np.zeros(len(params))
    
    obj = -np.log(ratio)
    
    # Numerical gradient (simple finite differences for reliability)
    grad = np.zeros(len(params))
    eps = 1e-6
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        h_plus = density_from_params(params_plus, N)
        ratio_plus = compute_ratio(h_plus)
        if ratio_plus < 1e-15:
            grad[i] = 0.0
        else:
            grad[i] = (-np.log(ratio_plus) - obj) / eps
    
    return obj, grad

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    # Start with the seed solution
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    # Try continuous density optimization
    np.random.seed(42)
    
    # Multiple random initializations for the continuous parameters
    num_starts = 20
    for start in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization of mixture parameters
        # mu in [-0.2, 0.2], sigma in [0.05, 0.15], weights in [-1, 1]
        if start == 0:
            # Start near a single peak
            params0 = np.array([0.0, 0.08, 0.0, 0.08, 0.0, -2.0])
        else:
            mu1 = np.random.uniform(-0.2, 0.2)
            mu2 = np.random.uniform(-0.2, 0.2)
            sigma1 = np.random.uniform(0.05, 0.15)
            sigma2 = np.random.uniform(0.05, 0.15)
            w1 = np.random.uniform(-1, 1)
            w2 = np.random.uniform(-1, 1)
            params0 = np.array([mu1, sigma1, mu2, sigma2, w1, w2])
        
        # Optimize using L-BFGS-B
        try:
            res = minimize(
                continuous_objective,
                params0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            params_opt = res.x
            h_opt = density_from_params(params_opt, N)
            sc = compute_ratio(h_opt)
            
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                
        except Exception as e:
            continue
    
    # Local search: try small perturbations of the continuous density
    if time.time() < deadline - 5.0:
        # Try shifting and scaling the best solution
        for shift in np.linspace(-0.1, 0.1, 21):
            if time.time() > deadline - 5.0:
                break
            for scale in np.linspace(0.5, 2.0, 16):
                if time.time() > deadline - 5.0:
                    break
                
                # Create shifted and scaled version
                x_orig = np.linspace(-0.25, 0.25, N)
                x_new = (x_orig - shift) * scale
                
                # Interpolate
                if np.all(x_new >= -0.25) and np.all(x_new <= 0.25):
                    h_shifted = np.interp(x_new, x_orig, best_h)
                    h_shifted = np.maximum(h_shifted, 0.0)
                    total = np.sum(h_shifted)
                    if total > 1e-15:
                        h_shifted = h_shifted / total
                        sc = compute_ratio(h_shifted)
                        if sc > best_score:
                            best_score = sc
                            best_h = h_shifted.copy()
    
    return list(best_h)
