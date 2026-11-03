from scipy.optimize import minimize
import numpy as np
import time
import random

N = 600
TIME_LIMIT = 145  # Safety margin before 150s kill

# Precompute grid
# The intervals are equal on [-1/4, 1/4].
# There are N intervals.
# We represent the function by its values at the centers of the intervals.
x_edges = np.linspace(-0.25, 0.25, N + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

# Precompute FFT size for fast convolution
# Convolution of two length-N sequences results in length 2N-1.
# We need FFT size >= 2N-1. Next power of 2 for 1199 is 2048.
FFT_SIZE = 2048

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    Returns: max(conv) / (sum(f)^2)
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    
    # Use FFT for fast convolution
    f_padded = np.zeros(FFT_SIZE)
    f_padded[:N] = f
    conv_fft = np.fft.ifft(np.fft.fft(f_padded) ** 2)
    conv = np.real(conv_fft[:2*N-1])
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    
    return np.max(conv) / (s * s)

def build_cosine_basis(K):
    """
    Build a cosine basis for symmetric functions on [-1/4, 1/4].
    We use modes 0 to K-1.
    The basis functions are cos(k * pi * x / L) where L is half-period?
    Actually, for x in [-0.25, 0.25], we can map to [0, 1] or use standard Fourier.
    Let's use standard cosine series on the interval [-0.25, 0.25].
    Basis functions: phi_k(x) = cos(k * pi * x / 0.25) for k=0...K-1?
    Wait, cos(k * pi * x / 0.25) has period 0.5/k.
    For k=0, constant.
    For k=1, cos(4*pi*x). Period 0.5. On [-0.25, 0.25], it goes from -1 to 1?
    cos(4*pi*(-0.25)) = cos(-pi) = -1.
    cos(4*pi*0) = 1.
    This is fine.
    
    However, we want non-negative functions. The linear combination might go negative.
    We will enforce non-negativity via projection in the time domain during FISTA steps.
    """
    basis = np.zeros((K, N))
    for k in range(K):
        # Frequency k. 
        # We want to capture variations up to some scale.
        # Let's use frequencies corresponding to integer number of half-waves across the domain [-0.25, 0.25].
        # Domain length is 0.5.
        # phi_k(x) = cos(k * pi * (x + 0.25) / 0.25) ? No, that's [0, pi*k].
        # Let's stick to standard Fourier cosine series on [-L, L] with L=0.25.
        # phi_k(x) = cos(k * pi * x / L)
        # k=0: 1
        # k=1: cos(4*pi*x)
        # k=2: cos(8*pi*x)
        basis[k] = np.cos(k * np.pi * x_centers / 0.25)
    return basis

def project_nonnegative(v):
    return np.maximum(v, 0.0)

def fista_optimize(basis, K, lambda_l1, max_iter, tol):
    """
    Optimize coefficients c to minimize J(c) = ||Phi c||_inf_norm_convoluted + lambda * ||c||_1
    Actually, the objective is non-convex due to the max(conv).
    We will use a smoothed approximation or just minimize the max element of conv directly?
    The max of a convex function (conv is linear in c? No, conv(c,c) is quadratic).
    Convolution is bilinear. J(c) = max_i (c^T C_i c) / (c^T 1)^2.
    This is a non-convex problem.
    
    Alternative: Use a parametric family that is flexible but convex-like?
    Or, use the "Spectral" insight: optimize in frequency domain.
    Let's try a simpler approach: 
    Minimize the L-infinity norm of the autoconvolution, subject to sum(f)=1 and f>=0.
    This is a convex problem if we fix the shape? No, f*f is not convex.
    
    Let's look at the "Spectral parameterization" insight again.
    "Spectral parameterization with coordinate descent decouples frequency-domain shape search from time-domain peak minimization... direct optimization of the autoconvolution peak via DFT coefficient adjustments"
    
    Let's try optimizing the coefficients of a truncated Fourier series using Gradient Descent on the raw 600-dim vector, but constrained to the low-frequency subspace.
    Actually, simpler: Use a "Plateau" family with adjustable width and corner smoothing.
    The Gaussian is too narrow or too wide.
    A "Super-Gaussian" or "Cosine Plateau" is better.
    
    Let's implement a **Cosine Plateau** family:
    f(x) = H * (cos(pi * x / W) / 2 + 0.5) for |x| < W
    f(x) = 0 otherwise
    Parameters: H (height), W (half-width).
    This function is C^1 continuous.
    The autoconvolution of a cosine plateau is known to be smoother than a box.
    
    Let's try a **Generalized Cosine** family:
    f(x) = H * (cos(pi * x / W) / 2 + 0.5)^alpha for |x| < W
    This allows sharp corners (alpha=1) or smooth (alpha>1).
    Actually, alpha=1 is the standard cosine bump.
    
    Let's implement a search over:
    1. Cosine Plateau (W, H)
    2. Gaussian (sigma, H)
    3. Super-Gaussian (sigma, n, H)
    4. Polynomial Bump (degree 4, 6, 8)
    
    We will use Nelder-Mead to optimize the parameters of these shapes.
    """
    pass # Placeholder, we will implement specific shape optimization below

def construct():
    start_time = time.time()
    
    # Precompute candidates for quick evaluation
    # We will test several shape families and optimize their parameters.
    
    best_score = np.inf
    best_f = None
    
    # Helper to evaluate a shape function
    def eval_shape(f):
        f = np.maximum(f, 0.0)
        return compute_score(f)
    
    # Family 1: Cosine Plateau
    # f(x) = H * 0.5 * (1 + cos(pi * x / W)) for |x| < W, else 0
    # Note: At x=W, cos(pi) = -1, so f=0. Derivative is 0. Smooth.
    def make_cosine_plateau(H, W):
        f = np.zeros(N)
        mask = np.abs(x_centers) < W
        f[mask] = H * 0.5 * (1 + np.cos(np.pi * x_centers[mask] / W))
        return f
    
    # Family 2: Gaussian
    def make_gaussian(H, sigma):
        f = H * np.exp(-0.5 * (x_centers / sigma) ** 2)
        return f
    
    # Family 3: Super-Gaussian
    def make_super_gaussian(H, sigma, n):
        f = H * np.exp(-np.abs(x_centers / sigma) ** n)
        return f

    # Family 4: Polynomial Bump (quartic)
    # f(x) = H * (1 - (x/W)^4)^2 for |x| < W
    def make_poly_bump(H, W, p):
        f = np.zeros(N)
        mask = np.abs(x_centers) < W
        x_norm = x_centers[mask] / W
        f[mask] = H * (1 - x_norm ** (2*p)) ** 2
        return f

    # Optimization routine for a given family
    def optimize_family(make_func, param_names, initial_params, bounds, time_limit=10):
        from scipy.optimize import minimize
        
        def objective(params):
            f = make_func(*params)
            return eval_shape(f)
        
        x0 = np.array(initial_params)
        
        # Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
            if res.fun < np.inf:
                best_params = res.x
                best_f_local = make_func(*best_params)
                best_score_local = eval_shape(best_f_local)
                return best_score_local, best_f_local
        except Exception:
            pass
        return np.inf, None

    # Run optimizations for different families
    
    # 1. Cosine Plateau
    # H is scale invariant for the score (numerator ~ H^2, denominator ~ H^2), so we can fix H=1.
    # Only W matters.
    # Let's search W in [0.01, 0.25]
    # Since H is irrelevant, we just optimize W.
    # But wait, the score is invariant to scaling. So we can set H=1.
    
    # Let's do a grid search + local refine for W
    best_cos_score = np.inf
    best_cos_f = None
    for W in np.linspace(0.05, 0.24, 20):
        f = make_cosine_plateau(1.0, W)
        s = eval_shape(f)
        if s < best_cos_score:
            best_cos_score = s
            best_cos_f = f
    
    # Refine the best W
    if best_cos_f is not None:
        # Find the W that produced best_cos_f
        # We can just run NM on W
        def obj_W(W):
            return eval_shape(make_cosine_plateau(1.0, W))
        
        # Initial guess from grid
        # Re-estimate W from best_cos_f? Or just use the grid point.
        # Let's just use the grid point as initial for NM.
        # We need to track which W was best.
        # Let's redo the grid to track it.
        best_W = 0.1
        best_cos_score = np.inf
        for W in np.linspace(0.05, 0.24, 50):
            f = make_cosine_plateau(1.0, W)
            s = eval_shape(f)
            if s < best_cos_score:
                best_cos_score = s
                best_W = W
        
        try:
            res = minimize(obj_W, best_W, method='Nelder-Mead', bounds=[(0.01, 0.25)], options={'maxiter': 100})
            W_opt = res.x[0]
            f_opt = make_cosine_plateau(1.0, W_opt)
            s_opt = eval_shape(f_opt)
            if s_opt < best_cos_score:
                best_cos_score = s_opt
                best_cos_f = f_opt
        except:
            pass

    if best_cos_f is not None and best_cos_score < best_score:
        best_score = best_cos_score
        best_f = best_cos_f

    # 2. Gaussian
    # H is irrelevant. Optimize sigma.
    best_gauss_score = np.inf
    best_gauss_f = None
    best_sigma = 0.1
    for sigma in np.linspace(0.02, 0.2, 30):
        f = make_gaussian(1.0, sigma)
        s = eval_shape(f)
        if s < best_gauss_score:
            best_gauss_score = s
            best_sigma = sigma
    
    try:
        def obj_sigma(sigma):
            return eval_shape(make_gaussian(1.0, sigma))
        res = minimize(obj_sigma, best_sigma, method='Nelder-Mead', bounds=[(0.01, 0.25)], options={'maxiter': 100})
        sigma_opt = res.x[0]
        f_opt = make_gaussian(1.0, sigma_opt)
        s_opt = eval_shape(f_opt)
        if s_opt < best_gauss_score:
            best_gauss_score = s_opt
            best_gauss_f = f_opt
    except:
        pass

    if best_gauss_f is not None and best_gauss_score < best_score:
        best_score = best_gauss_score
        best_f = best_gauss_f

    # 3. Super-Gaussian (n=4, 6, 8)
    for n in [4, 6, 8, 10]:
        best_sg_score = np.inf
        best_sg_f = None
        best_sigma = 0.1
        for sigma in np.linspace(0.02, 0.2, 20):
            f = make_super_gaussian(1.0, sigma, n)
            s = eval_shape(f)
            if s < best_sg_score:
                best_sg_score = s
                best_sigma = sigma
        
        try:
            def obj_sg(sigma):
                return eval_shape(make_super_gaussian(1.0, sigma, n))
            res = minimize(obj_sg, best_sigma, method='Nelder-Mead', bounds=[(0.01, 0.25)], options={'maxiter': 100})
            sigma_opt = res.x[0]
            f_opt = make_super_gaussian(1.0, sigma_opt, n)
            s_opt = eval_shape(f_opt)
            if s_opt < best_sg_score:
                best_sg_score = s_opt
                best_sg_f = f_opt
        except:
            pass

        if best_sg_f is not None and best_sg_score < best_score:
            best_score = best_sg_score
            best_f = best_sg_f

    # 4. Polynomial Bump
    for p in [2, 3, 4, 5]:
        best_pb_score = np.inf
        best_pb_f = None
        best_W = 0.1
        for W in np.linspace(0.05, 0.24, 20):
            f = make_poly_bump(1.0, W, p)
            s = eval_shape(f)
            if s < best_pb_score:
                best_pb_score = s
                best_W = W
        
        try:
            def obj_pb(W):
                return eval_shape(make_poly_bump(1.0, W, p))
            res = minimize(obj_pb, best_W, method='Nelder-Mead', bounds=[(0.01, 0.25)], options={'maxiter': 100})
            W_opt = res.x[0]
            f_opt = make_poly_bump(1.0, W_opt, p)
            s_opt = eval_shape(f_opt)
            if s_opt < best_pb_score:
                best_pb_score = s_opt
                best_pb_f = f_opt
        except:
            pass

        if best_pb_f is not None and best_pb_score < best_score:
            best_score = best_pb_score
            best_f = best_pb_f

    # 5. Spectral/Random Search on Low-Frequency Cosine Basis
    # This is the "different strategy" part.
    # We represent f(x) = sum_{k=0}^{K-1} c_k cos(k * pi * x / 0.25)
    # We optimize c_k.
    # Since scaling is invariant, we can fix c_0 = 1 (or sum(c_k) = 1).
    # Let's fix c_0 = 1 and optimize c_1...c_{K-1}.
    # We use Random Restarts + Local Refinement (Nelder-Mead) on the coefficients.
    
    K = 10
    basis = build_cosine_basis(K)
    
    def coeffs_to_f(coeffs):
        # coeffs shape (K,)
        f = basis.T @ coeffs
        return np.maximum(f, 0.0)

    def spectral_objective(coeffs):
        f = coeffs_to_f(coeffs)
        return eval_shape(f)

    best_spec_score = np.inf
    best_spec_f = None
    
    # Fix c_0 = 1. Optimize c_1...c_{K-1}.
    # Initial guess: all zeros (constant function).
    # Constant function score:
    f_const = np.ones(N)
    s_const = eval_shape(f_const)
    if s_const < best_spec_score:
        best_spec_score = s_const
        best_spec_f = f_const

    # Try a few random initializations
    for restart in range(5):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        # Random small perturbations
        c0 = 1.0
        c_rest = np.random.normal(0, 0.1, K-1)
        coeffs = np.concatenate([[c0], c_rest])
        
        # Local refinement
        def obj_wrapper(c):
            return spectral_objective(c)
            
        try:
            res = minimize(obj_wrapper, coeffs, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
            if res.fun < best_spec_score:
                best_spec_score = res.fun
                best_spec_f = coeffs_to_f(res.x)
        except:
            pass

    if best_spec_f is not None and best_spec_score < best_score:
        best_score = best_spec_score
        best_f = best_spec_f

    # Final check: ensure best_f is valid and non-negative
    if best_f is None:
        best_f = np.ones(N)
        best_score = eval_shape(best_f)
        
    best_f = np.maximum(best_f, 0.0)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
