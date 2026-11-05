import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 400

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = 2 * N - 1
    H = rfft(h, n=n)
    c = irfft(H * H, n=n)
    return c

def _soft_max(x, beta):
    """Differentiable approximation to max using log-sum-exp."""
    m = np.max(x)
    x_shifted = x - m
    # Clip to avoid overflow
    x_shifted = np.clip(x_shifted, -50, 50)
    sum_exp = np.sum(np.exp(beta * x_shifted))
    if sum_exp == 0:
        return m
    return m + (1.0 / beta) * np.log(sum_exp)

def _evaluate_exact(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _objective(params, beta):
    """Soft-max objective for full h vector."""
    h = params
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    if not np.isfinite(val):
        return 1e10
    return val

def _gradient(params, beta):
    """Numerical gradient for the soft-max objective."""
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(N):
        params[i] += eps
        f_plus = _objective(params, beta)
        params[i] -= 2 * eps
        f_minus = _objective(params, beta)
        params[i] += eps
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # should be 2.0
    
    def try_h(h):
        nonlocal best_h, best_val
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        val = _evaluate_exact(h)
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
        return val
    
    # Phase 1: Generate diverse initial candidates
    # Strategy: Create signals with small sums to exploit the denominator
    candidates = []
    
    # 1a: Random signals with small mean
    for _ in range(20):
        h = rng.standard_normal(N)
        # Scale to have small sum
        mean_h = np.mean(h)
        h = h - mean_h + rng.uniform(-0.01, 0.01)
        candidates.append(h)
    
    # 1b: Random walk with drift (creates near-zero-sum naturally)
    for seed in range(10):
        rng_local = np.random.default_rng(seed)
        drift = rng_local.uniform(0.01, 0.5)
        noise = rng_local.uniform(0.5, 2.0)
        steps = rng_local.standard_normal(N) * noise - drift
        h = np.cumsum(steps)
        h = h - np.mean(h) + rng_local.uniform(-0.01, 0.01)
        candidates.append(h)
    
    # 1c: Sparse signals (mostly zero, few non-zero entries)
    for _ in range(10):
        h = np.zeros(N)
        n_nonzero = rng.integers(5, 30)
        indices = rng.choice(N, size=n_nonzero, replace=False)
        h[indices] = rng.standard_normal(n_nonzero)
        # Add small offset to avoid exact zero sum
        h += rng.uniform(-0.01, 0.01)
        candidates.append(h)
    
    # 1d: Bimodal signals (two clusters of different values)
    for _ in range(10):
        h = np.zeros(N)
        split = rng.integers(N // 3, 2 * N // 3)
        h[:split] = rng.uniform(0.5, 2.0)
        h[split:] = rng.uniform(-2.0, -0.5)
        # Adjust to have small sum
        current_sum = np.sum(h)
        if abs(current_sum) > 0.1:
            h -= current_sum / N
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        try_h(h)
    
    # Phase 2: L-BFGS-B optimization from best candidate
    # Use multi-stage beta annealing for better convergence
    betas = [10.0, 50.0, 200.0, 500.0, 1000.0]
    
    current_h = best_h.copy()
    
    for beta in betas:
        if time.time() > deadline - 30.0:
            break
        
        # Try multiple initial points around current best
        for restart in range(3):
            if time.time() > deadline - 25.0:
                break
            
            if restart == 0:
                init_h = current_h.copy()
            else:
                # Perturb the current best
                init_h = current_h + rng.standard_normal(N) * 0.1 * np.std(current_h)
            
            # Ensure sum is not too small
            s = np.sum(init_h)
            if abs(s) < 0.01:
                init_h += 0.01
            
            try:
                res = minimize(
                    lambda p, b=beta: _objective(p, b),
                    init_h,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
                )
                
                if res.success or res.fun < best_val:
                    val_opt = _evaluate_exact(res.x)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = res.x.copy()
                        current_h = res.x.copy()
                        
            except Exception:
                pass
    
    # Phase 3: Local refinement with coordinate descent
    if time.time() < deadline - 15.0:
        h_refine = best_h.copy()
        improved = True
        max_iter = 50
        
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            for i in range(0, N, 4):  # Step by 4 for speed
                if time.time() > deadline - 10.0:
                    break
                
                original = h_refine[i]
                best_local = h_refine[i]
                best_local_val = _evaluate_exact(h_refine)
                
                # Try several perturbations
                for delta in [-0.5, -0.1, 0.1, 0.5, 1.0, -1.0]:
                    h_refine[i] = original + delta * np.std(h_refine)
                    val = _evaluate_exact(h_refine)
                    if val < best_local_val:
                        best_local_val = val
                        best_local = h_refine[i]
                
                h_refine[i] = best_local
                if best_local != original:
                    improved = True
        
        val_refine = _evaluate_exact(h_refine)
        if val_refine < best_val:
            best_val = val_refine
            best_h = h_refine.copy()
    
    # Phase 4: Final evaluation and return
    final_val = _evaluate_exact(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
