import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 600
dx = 1.0 / (2 * N)
x = np.linspace(-0.25, 0.25, N, endpoint=False)

def compute_raw_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of nonnegative step heights.
    Score = max(autoconv) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-12):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-12:
        return np.inf
    
    # Compute autoconvolution using FFT
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    
    return np.max(conv) / (s * s)

def generate_gaussian(sigma):
    return np.exp(-0.5 * (x / sigma)**2)

def generate_super_gaussian(beta):
    # Scale x to be around 1 for typical beta
    # x is in [-0.25, 0.25]
    # f = exp(-|x/scale|^beta)
    # We want the width to be comparable to the domain
    # Let's define f = exp(-|x/0.25|^beta)
    return np.exp(-np.abs(x / 0.25)**beta)

def generate_raised_cosine(width):
    # width is fraction of domain [-0.25, 0.25]
    # Support [-width/2, width/2]
    half_w = 0.25 * width / 2.0
    f = np.zeros(N)
    mask = np.abs(x) < half_w
    f[mask] = np.cos(np.pi * x[mask] / (2 * half_w))**2
    return f

def scan_parametric():
    best_f = None
    best_val = np.inf
    
    # Gaussian
    sigmas = np.linspace(0.01, 0.25, 50)
    for sig in sigmas:
        f = generate_gaussian(sig)
        val = compute_raw_score(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
            
    # Super-Gaussian
    betas = np.linspace(1, 10, 50)
    for b in betas:
        f = generate_super_gaussian(b)
        val = compute_raw_score(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
            
    # Raised Cosine
    widths = np.linspace(0.1, 1.0, 50)
    for w in widths:
        f = generate_raised_cosine(w)
        val = compute_raw_score(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
            
    return best_f, best_val

def refine_with_lbfgsb(f_init, start_time, max_time):
    """
    Refine the step heights using L-BFGS-B.
    """
    if time.time() - start_time > max_time - 5:
        return f_init, compute_raw_score(f_init)
    
    def objective(x):
        f = np.maximum(x, 0.0)
        s = np.sum(f)
        if s < 1e-12:
            return 1e10
        f_norm = f / s
        return compute_raw_score(f_norm)
        
    # We don't have a good gradient for the raw score with max.
    # We can use smooth max for gradient.
    def smooth_objective_and_grad(x, beta=100.0):
        f = np.maximum(x, 0.0)
        s = np.sum(f)
        if s < 1e-12:
            return 1e10, np.zeros_like(x)
        f_norm = f / s
        
        n_fft = 2 * N
        F = np.fft.rfft(f_norm, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
        
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros_like(x)
            
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        if sum_exp < 1e-15:
            return 0.0, np.zeros_like(x)
        weights = exp_shifted / sum_exp
        
        W = np.fft.rfft(weights, n=n_fft)
        F_full = np.fft.rfft(f_norm, n=n_fft)
        grad_conv = 2.0 * np.fft.irfft(W * np.conj(F_full), n=n_fft)[:N]
        
        # grad of f_norm wrt f is 1/s - f/s^2
        # grad of score wrt f = grad_conv * (1/s - f/s^2)
        grad_f = grad_conv * (1.0/s - f/(s*s))
        
        val = max_c / (s * s)
        
        return val, grad_f

    f = f_init.copy()
    s = np.sum(f)
    if s < 1e-12:
        return f, compute_raw_score(f)
    f = f / s # Normalize
    
    best_val = compute_raw_score(f)
    best_f = f.copy()
    
    try:
        res = minimize(
            smooth_objective_and_grad, 
            f, 
            jac=True,
            method='L-BFGS-B', 
            bounds=[(0, None)]*N,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 5000}
        )
        if res.success or res.fun < best_val:
            f_res = res.x
            f_res = np.maximum(f_res, 0.0)
            s_res = np.sum(f_res)
            if s_res > 1e-12:
                f_res = f_res / s_res
                val_res = compute_raw_score(f_res)
                if val_res < best_val:
                    best_val = val_res
                    best_f = f_res
    except Exception as e:
        pass
        
    return best_f, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Parametric scan
    f_best, val_best = scan_parametric()
    
    # 2. Refine
    if time.time() < deadline - 10:
        f_best, val_best = refine_with_lbfgsb(f_best, start_time, deadline)
        
    # 3. Try a few random perturbations of the best found
    # This might help escape local minima if L-BFGS-B got stuck
    if time.time() < deadline - 10:
        for i in range(5):
            if time.time() > deadline - 5:
                break
            noise = np.random.normal(0, 0.01 * np.max(f_best), N)
            f_pert = np.maximum(f_best + noise, 0)
            s_pert = np.sum(f_pert)
            if s_pert > 1e-12:
                f_pert = f_pert / s_pert
                val_pert = compute_raw_score(f_pert)
                if val_pert < val_best:
                    f_best, val_best = refine_with_lbfgsb(f_pert, start_time, deadline)
                    
    return list(f_best)

if __name__ == "__main__":
    f = construct()
    print(f"Length: {len(f)}")
    print(f"Score: {compute_raw_score(np.array(f))}")
