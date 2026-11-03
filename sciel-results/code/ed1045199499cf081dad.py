import numpy as np
import time
from scipy.optimize import minimize, linprog

def _autoconv(f):
    """Compute autoconvolution (convolution f*f) using FFT."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def time_remaining():
        return deadline - time.time()

    # Strategy: Spectral representation + L2 projection + L-BFGS-B
    # Represent f as a sum of low-frequency cosine modes.
    # f_k = sum_{j=0}^{M-1} a_j * cos(pi * j * k / n)
    # This constrains f to be smooth, which is often optimal for minimizing max autoconvolution
    # (since smooth functions have smoother autoconvolutions, potentially lower peaks relative to mass).
    # We optimize the coefficients 'a' using L-BFGS-B, which is efficient for smooth objectives.
    # Since the objective is scale-invariant, we fix the sum of f to be 1 (or L2 norm to 1)
    # by normalizing f after computing it from 'a'.
    # Actually, L-BFGS-B works on the parameter space. If we normalize f inside the objective,
    # the gradient w.r.t 'a' is correct up to a scale factor, which L-BFGS-B can handle.
    
    # Let's try optimizing the coefficients of a cosine basis.
    # Number of modes M.
    # Too many modes -> high dimensional, less smooth.
    # Too few modes -> might not be flexible enough.
    # Let's start with M=20.
    
    M = 20
    # Basis functions: cos(pi * j * k / n) for j=0..M-1
    # k = 0..n-1
    k = np.arange(n)
    # J = 0..M-1
    J = np.arange(M)
    # Basis matrix B of shape (M, n)
    # B[j, k] = cos(pi * j * k / n)
    # Note: j=0 gives constant 1.
    # We want f = B.T @ a
    # To ensure f is non-negative, we can use softplus or max(0, .) in the objective,
    # but gradients of max are problematic.
    # Alternatively, we can parameterize a such that f is positive?
    # Or just use a soft penalty for negative values?
    # Or, since cosine sums can be negative, maybe use a different basis?
    # Actually, for minimizing max autoconvolution, the optimal f is likely positive and smooth.
    # The constant mode j=0 will dominate to keep it positive.
    # Let's just use the raw coefficients and rely on the optimizer to keep the sum positive.
    # We will normalize f by its sum. If sum <= 0, return a large penalty.
    
    def objective(a):
        # a is shape (M,)
        f = np.dot(a, np.cos(np.pi * np.outer(J, k) / n))
        # f is shape (n,)
        
        # Ensure non-negativity?
        # If f has negative parts, the autoconvolution might be weird, and sum might be small.
        # The problem requires f >= 0.
        # We can project f onto non-negative reals? No, that breaks the smoothness of the objective w.r.t a.
        # Better: add a penalty for negative values?
        # Or, simply clip f to 0? Clipping makes the objective non-smooth.
        # Let's try: if min(f) < 0, return a penalty proportional to the magnitude of negativity.
        # Or, we can restrict the search space of 'a' to ensure positivity?
        # That's hard.
        
        # Let's try a different approach:
        # Use a basis that is inherently non-negative? E.g., B-splines?
        # Or just use the cosine basis and hope the optimizer stays in the positive region.
        # The constant term a_0 can be made large enough to offset the rest.
        
        # Let's check for negativity.
        min_f = np.min(f)
        if min_f < -1e-6:
            # Penalty
            penalty = 100.0 * np.sum(np.maximum(-f, 0)**2)
            return penalty + 100.0 # Large offset
            
        # If f is mostly positive, normalize
        sum_f = np.sum(f)
        if sum_f < 1e-6:
            return 100.0
            
        f_norm = f / sum_f
        
        # Compute score
        conv = _autoconv(f_norm)
        max_conv = np.max(conv)
        
        return max_conv

    # Initial guess: a_0 = 1, others 0 (constant function)
    a0 = np.zeros(M)
    a0[0] = 1.0
    
    # Run L-BFGS-B
    # Bounds for a: [-10, 10]
    bounds = [(-10.0, 10.0)] * M
    
    best_a = a0
    best_score = _compute_score(np.ones(n)) # Constant function score
    
    # Try multiple restarts with different M or different initial a?
    # Let's just run one L-BFGS-B with a reasonable budget.
    
    try:
        res = minimize(
            objective,
            a0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_score:
            best_a = res.x
            # Compute actual score with clipping/normalization
            f_candidate = np.dot(best_a, np.cos(np.pi * np.outer(J, k) / n))
            f_candidate = np.maximum(f_candidate, 0.0)
            sum_f = np.sum(f_candidate)
            if sum_f > 1e-6:
                f_candidate = f_candidate / sum_f
                score = _compute_score(f_candidate)
                if score < best_score:
                    best_score = score
                    # Store best_f
                    best_f = f_candidate
                else:
                    best_f = f_candidate
            else:
                best_f = np.ones(n)
        else:
            best_f = np.ones(n)
            
    except Exception as e:
        best_f = np.ones(n)
        best_score = _compute_score(best_f)

    # If time remains, try a higher M or different optimization?
    # Let's try a local refinement on the full 600-dim vector using the best_f as start.
    # But L-BFGS-B on 600 dims might be slow.
    # Instead, let's try a few random restarts of the spectral method with different M.
    
    if time_remaining() > 30.0:
        for M_test in [10, 30, 50]:
            if time_remaining() < 10.0:
                break
            
            J_test = np.arange(M_test)
            def objective_test(a):
                f = np.dot(a, np.cos(np.pi * np.outer(J_test, k) / n))
                min_f = np.min(f)
                if min_f < -1e-6:
                    penalty = 100.0 * np.sum(np.maximum(-f, 0)**2)
                    return penalty + 100.0
                sum_f = np.sum(f)
                if sum_f < 1e-6:
                    return 100.0
                f_norm = f / sum_f
                conv = _autoconv(f_norm)
                max_conv = np.max(conv)
                return max_conv
            
            a0_test = np.zeros(M_test)
            a0_test[0] = 1.0
            bounds_test = [(-10.0, 10.0)] * M_test
            
            try:
                res_test = minimize(
                    objective_test,
                    a0_test,
                    method='L-BFGS-B',
                    bounds=bounds_test,
                    options={'maxiter': 100, 'maxfun': 500}
                )
                
                if res.fun < best_score:
                    f_candidate = np.dot(res_test.x, np.cos(np.pi * np.outer(J_test, k) / n))
                    f_candidate = np.maximum(f_candidate, 0.0)
                    sum_f = np.sum(f_candidate)
                    if sum_f > 1e-6:
                        f_candidate = f_candidate / sum_f
                        score = _compute_score(f_candidate)
                        if score < best_score:
                            best_score = score
                            best_f = f_candidate
            except:
                pass

    # Final safety: ensure non-negative and not zero
    best_f = np.maximum(best_f, 0.0)
    if np.all(best_f == 0):
        best_f = np.ones(n)
        
    return list(best_f)
