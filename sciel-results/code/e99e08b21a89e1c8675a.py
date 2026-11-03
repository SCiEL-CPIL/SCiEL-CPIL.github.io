"""
Task: Minimize the normalized maximum of autoconvolution of a nonnegative step function
with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Parametric representation using a cosine basis to ensure symmetry and smoothness.
2. Optimization of the coefficients in the cosine basis using L-BFGS-B.
3. The objective is the max of the autoconvolution, normalized by (sum f)^2.
4. Non-negativity is enforced by projecting onto the non-negative orthant in the time domain
   after each gradient step, or by using a softplus-like transformation if needed, but
   simple projection is often sufficient for this type of problem.
5. Local coordinate descent polish on the time-domain values to fine-tune.
"""

import time
import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of nonnegative values.
    Returns: max(autoconv(f)) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return 1e10
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s <= 1e-15:
        return 1e10
    # Normalize f to sum to 1 to make the problem scale-invariant
    # The score is scale-invariant: max((cf)*(cf)) / (sum(cf))^2 = max(f*f)/(sum f)^2
    f_norm = f / s
    # Compute autoconvolution via FFT
    F = fft(f_norm)
    conv_F = F * F
    conv_t = ifft(conv_F).real
    return np.max(conv_t)

def smooth_max(x, k=100):
    """
    Smooth approximation of the max function.
    log-sum-exp trick: (1/k) * log(sum(exp(k*x)))
    """
    x_max = np.max(x)
    if x_max > 1e6:
        return x_max
    return x_max + np.log(np.sum(np.exp(k * (x - x_max)))) / k

def cosine_basis(N):
    """
    Generate cosine basis functions for symmetric step functions.
    We represent f[i] = sum_j c_j * cos(2*pi*i*j/N) ?
    Actually, for a symmetric function on [-1/4, 1/4], we can use:
    f(x) = sum_{k=0}^{K} a_k * cos(2 * pi * k * x / 0.5)
    But we are on a grid.
    Let's use a discrete cosine transform (DCT) like basis or just simple cosines.
    Since the domain is symmetric, we can parameterize the first N/2+1 values?
    No, the function is defined on N intervals.
    Let's use a basis of cosines: phi_k(i) = cos(2 * pi * k * i / N)
    This ensures periodicity, which might be an artifact, but it's a valid basis.
    Better: use polynomials or cosines defined on the physical domain [-0.25, 0.25].
    x_i = -0.25 + (i + 0.5) * (0.5 / N)  # Center of interval i
    Basis: phi_k(x) = cos(2 * pi * k * x) for k=0..K.
    This allows smooth, symmetric (if we only use even modes? No, cos is even) functions.
    cos(2*pi*k*x) is even around 0.
    So f(x) = sum a_k cos(2*pi*k*x) is automatically even.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N  # Centers of intervals
    # Actually, linspace(-0.25, 0.25, N, endpoint=False) gives left edges.
    # Centers are left_edge + width/2.
    width = 0.5 / N
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + width / 2
    
    # Number of basis functions
    K = 20
    basis = np.zeros((N, K + 1))
    for k in range(K + 1):
        basis[:, k] = np.cos(2 * np.pi * k * x)
    return basis, x

def parametric_to_step(coeffs, basis):
    """
    Convert coefficients to step function values.
    coeffs: array of length K+1
    Returns: array of length N, non-negative.
    """
    f = basis @ coeffs
    # Ensure non-negativity
    f = np.maximum(f, 0)
    return f

def objective(coeffs, basis):
    """
    Objective function for optimization.
    """
    f = parametric_to_step(coeffs, basis)
    return compute_score(f)

def local_refinement(f, max_iter=500, time_limit=5.0):
    """
    Coordinate descent on the time-domain values.
    """
    start_time = time.time()
    f = f.copy()
    current_score = compute_score(f)
    
    # Simple coordinate descent
    # Step size can be adaptive
    step_size = 1e-4
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            # Try increasing
            f_new = f.copy()
            f_new[i] += step_size
            s_new = compute_score(f_new)
            if s_new < current_score:
                f = f_new
                current_score = s_new
                improved = True
                continue
            
            # Try decreasing
            if f[i] > 1e-8:
                f_new = f.copy()
                f_new[i] -= step_size
                s_new = compute_score(f_new)
                if s_new < current_score:
                    f = f_new
                    current_score = s_new
                    improved = True
    
    return f, current_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # 1. Generate Basis
    basis, x = cosine_basis(N)
    K = basis.shape[1] - 1
    
    # 2. Initial Coefficients
    # Start with a Gaussian-like profile.
    # Approximate a Gaussian with cosines.
    # Or just start with a broad peak.
    # a_0 controls the mean. a_k controls the oscillations.
    # Let's initialize with a Gaussian in time domain and project onto the basis.
    sigma = 0.1
    h_init = np.exp(-x**2 / (2 * sigma**2))
    # Least squares fit to the basis
    coeffs_init, residuals, rank, sv = np.linalg.lstsq(basis, h_init, rcond=None)
    
    # Ensure the resulting f is non-negative and normalized
    f_init = parametric_to_step(coeffs_init, basis)
    
    # 3. Optimization
    # We optimize the coefficients.
    # To handle non-negativity, we rely on the projection in parametric_to_step.
    # However, this makes the objective non-smooth (due to max and clipping).
    # L-BFGS-B might struggle.
    # Alternative: Optimize in time domain directly with L-BFGS-B?
    # The dimension is 600. L-BFGS-B can handle that.
    # Let's try time-domain optimization directly.
    
    def time_domain_obj(f):
        return compute_score(f)
    
    def time_domain_grad(f):
        # Gradient of max(f*f) / (sum f)^2
        # This is tricky. Let's use numerical gradient or a simpler approach.
        # Actually, let's stick to the parametric approach but use a smoother objective.
        # Or use SLSQP with constraints.
        
        # Let's try a different strategy:
        # The problem is convex in the time domain?
        # min max_k (f*f)_k s.t. f >= 0, sum f = 1
        # This is a convex problem.
        # We can use projected subgradient or similar.
        # But L-BFGS-B doesn't handle max directly well without smoothing.
        
        # Let's use the smooth_max approximation in the parametric space.
        pass
        
    # Let's try optimizing the time-domain vector directly using L-BFGS-B
    # with a smooth approximation of the max.
    
    def smooth_obj(f):
        s = np.sum(f)
        if s <= 1e-15:
            return 1e10
        f_norm = f / s
        F = fft(f_norm)
        conv_F = F * F
        conv_t = ifft(conv_F).real
        return smooth_max(conv_t, k=50)
        
    # Initialize with the Gaussian
    f_current = h_init.copy()
    f_current = np.maximum(f_current, 0)
    f_current /= np.sum(f_current)
    
    # Optimization in time domain
    # Constraints: f >= 0, sum f = 1
    # L-BFGS-B with bounds [0, inf]
    # We don't strictly need sum f = 1 because the objective is scale-invariant,
    # but it helps numerical stability.
    
    result = minimize(
        smooth_obj,
        f_current,
        method='L-BFGS-B',
        bounds=[(0, None)] * N,
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    f_opt = result.x
    f_opt = np.maximum(f_opt, 0)
    f_opt /= np.sum(f_opt)
    
    current_score = compute_score(f_opt)
    
    # 4. Local Refinement
    if time.time() < deadline - 10:
        f_opt, current_score = local_refinement(f_opt, max_iter=200, time_limit=5)
        
    # 5. Multi-start
    # Try different initializations
    inits = [
        np.exp(-x**2 / (2 * 0.05**2)),
        np.exp(-x**2 / (2 * 0.15**2)),
        np.ones(N) * 0.1, # Flat
        np.exp(-np.abs(x) / 0.1) # Exponential
    ]
    
    for h_init in inits:
        if time.time() > deadline - 10:
            break
        h_try = np.maximum(h_init, 0)
        s_try = np.sum(h_try)
        if s_try > 0:
            h_try /= s_try
            res_try = minimize(
                smooth_obj,
                h_try,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            f_try = res_try.x
            f_try = np.maximum(f_try, 0)
            s_try = np.sum(f_try)
            if s_try > 0:
                f_try /= s_try
                score_try = compute_score(f_try)
                if score_try < current_score:
                    current_score = score_try
                    f_opt = f_try
                    # Refine
                    if time.time() < deadline - 5:
                        f_opt, current_score = local_refinement(f_opt, max_iter=100, time_limit=3)

    return list(f_opt)

if __name__ == "__main__":
    res = construct()
    print("Constructed:", len(res))
    print("Score:", compute_score(np.array(res)))
