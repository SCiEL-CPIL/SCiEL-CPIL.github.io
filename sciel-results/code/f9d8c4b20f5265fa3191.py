"""
Optimized program for uncertainty inequality task.
Strategy: Chebyshev collocation for exact Fourier transform + Differential Evolution.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import eval_hermite
import time

def _setup_chebyshev(N=256, L=12.0):
    """
    Set up Chebyshev collocation on [-L, L].
    Returns:
    x: grid points
    D: differentiation matrix (not directly needed for FT, but good for verification)
    w: quadrature weights for integration
    """
    # Chebyshev nodes in [-1, 1]
    t = np.cos(np.pi * (np.arange(N) + 0.5) / N)
    # Map to [-L, L]
    x = L * t
    
    # Weights for Chebyshev-Gauss quadrature
    # w_j = (pi/N) * c_j * (1 - t_j^2)^0.5 ? No, standard weights are:
    # w_j = pi/N for j=1..N-1, w_1 = w_N = pi/(2N) (for integral -1 to 1)
    # But we are integrating over [-L, L].
    # Integral_{-L}^L f(x) dx = L * Integral_{-1}^1 f(Lt) dt
    # Using Chebyshev nodes t_j.
    
    # Actually, for spectral accuracy, we can just use the nodes and standard weights.
    # The weights for Chebyshev-Gauss quadrature on [-1,1] are:
    w_t = np.full(N, np.pi / N)
    w_t[0] = np.pi / (2 * N)
    w_t[-1] = np.pi / (2 * N)
    
    # Map to [-L, L]: dx = L dt
    w = L * w_t
    
    # Sort x in increasing order for easier handling? 
    # t is decreasing (cos from 0 to pi). So x is decreasing from L to -L.
    # Let's reverse to make x increasing.
    x = x[::-1]
    w = w[::-1]
    t = t[::-1]
    
    return x, w, t

def _compute_bound_cheb(a0, a4, a8, x, w, t, N=256, L=12.0):
    """
    Compute the uncertainty bound C4 using Chebyshev nodes.
    f(x) = g(x) * exp(-x^2/4)
    g(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    a12 is fixed by g(0)=0.
    """
    # Compute a12
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Evaluate Hermite polynomials at x
    # Using recurrence or direct evaluation. Direct is fine for small N.
    # H_n(x)
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = x**12 - 42*x**10 + 840*x**8 - 7560*x**6 + 30240*x**4 - 45360*x**2 + 10395
    
    g = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    # Gaussian factor
    gauss = np.exp(-x**2 / 4.0)
    f = g * gauss
    
    # Compute ||f||_1 and ||f||_2 using quadrature
    abs_f = np.abs(f)
    f_sq = f**2
    
    norm1 = np.sum(abs_f * w)
    norm2_sq = np.sum(f_sq * w)
    
    if norm2_sq < 1e-30:
        return 1e10
        
    norm2 = np.sqrt(norm2_sq)
    
    # Compute Fourier Transform F(k)
    # F(k) = integral_{-L}^{L} f(x) e^{-ikx} dx
    # We need to estimate ||F||_1 and ||F||_2.
    # Since f is band-limited-ish (Gaussian), we can evaluate F(k) on a grid of k.
    # However, to be "exact" in the spectral sense, we can compute the coefficients
    # of the Fourier series if we were on a periodic domain, but here we have a compact support approximation.
    
    # Better approach: Evaluate F(k) at a set of points k_j.
    # To get ||F||_1 and ||F||_2, we need to integrate |F(k)| and |F(k)|^2 over k.
    # We can use a symmetric grid for k.
    
    # Let's use a grid for k.
    # The function decays as exp(-x^2/4), so it's smooth.
    # We can evaluate F(k) exactly for any k using the same quadrature:
    # F(k) = sum_j f(x_j) e^{-i k x_j} w_j
    
    # How many k points?
    # To capture the structure, let's use a similar number of points.
    # Let's pick k_max such that we cover the significant part of the spectrum.
    # For exp(-x^2/4), the FT is proportional to exp(-k^2). 
    # Significant range is roughly k in [-8, 8].
    
    n_k = 512
    k_max = 10.0
    k = np.linspace(-k_max, k_max, n_k)
    dk = k[1] - k[0]
    
    # Compute F(k) for all k
    # F[k_idx] = sum_j f[j] * exp(-1j * k[idx] * x[j]) * w[j]
    # This is an exponential sum.
    # Can be vectorized: exp_matrix = exp(-1j * np.outer(k, x))
    # But n_k * N might be large (512 * 256 = 131k). That's fine.
    
    # To save memory and time, we can do it in chunks or just use the matrix.
    # 131k complex numbers is small.
    
    # Construct the kernel
    # K[i, j] = exp(-1j * k[i] * x[j])
    # F[i] = sum_j K[i, j] * f[j] * w[j]
    
    # Let's compute this.
    # Note: x is sorted increasing.
    
    # Optimization: Precompute f*w
    fw = f * w
    
    # Compute F
    # Using broadcasting
    # k[:, None] * x[None, :]
    # This creates a 512x256 matrix.
    
    # Check time: 512*256 exp operations. ~130k. Very fast.
    
    kx = np.outer(k, x)
    kernel = np.exp(-1j * kx)
    
    F = kernel @ fw
    
    # Compute norms of F
    # ||F||_1 = integral |F(k)| dk approx sum |F(k)| dk
    # ||F||_2 = integral |F(k)|^2 dk approx sum |F(k)|^2 dk
    
    abs_F = np.abs(F)
    F_sq = F.real**2 + F.imag**2
    
    F_norm1 = np.sum(abs_F) * dk
    F_norm2_sq = np.sum(F_sq) * dk
    
    if F_norm2_sq < 1e-30:
        return 1e10
        
    F_norm2 = np.sqrt(F_norm2_sq)
    
    # Bound = (||f||_1 * ||F||_1) / (||f||_2 * ||F||_2)
    bound = (norm1 * F_norm1) / (norm2 * F_norm2)
    
    return bound

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Setup Chebyshev grid
    # N=256 is good for accuracy without being too slow.
    # L=12.0 covers the Gaussian decay well (exp(-144/4) = exp(-36) ~ 2e-16)
    N = 256
    L = 12.0
    x, w, t = _setup_chebyshev(N, L)
    
    # Define objective for optimizer
    def objective(params):
        a0, a4, a8 = params
        try:
            b = _compute_bound_cheb(a0, a4, a8, x, w, t, N, L)
            if not np.isfinite(b):
                return 1e10
            return b
        except:
            return 1e10
            
    # Differential Evolution is a global optimizer.
    # Bounds for coefficients.
    # The problem is scale invariant, so we can bound the magnitude.
    # Let's bound each coefficient to [-10, 10].
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    best_bound = 1e10
    best_coeffs = np.array([1.0, 0.0, 0.0])
    
    # Run Differential Evolution
    # maxiter: number of generations. 
    # popsize: population size.
    # We have ~170 seconds. 
    # Each evaluation is ~1ms (512*256 exp + matmul). 
    # So we can afford ~100k evaluations.
    # DE with popsize=20, maxiter=1000 -> 200k evaluations. Might be too many if slow.
    # Let's try popsize=15, maxiter=200. 15*200 = 3000 evals. ~3 seconds.
    # We can do more.
    
    # Let's run a first pass.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=20, 
            tol=1e-8, 
            polish=True, 
            init='sobol'
        )
        
        if result.fun < best_bound:
            best_bound = result.fun
            best_coeffs = result.x
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Check time
    if time.time() > deadline - 30:
        return best_coeffs
        
    # Run a second, more refined DE or local search from the best point
    # Use Nelder-Mead for polish if time permits
    
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        try:
            result2 = minimize(
                objective, 
                best_coeffs, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result2.fun < best_bound:
                best_bound = result2.fun
                best_coeffs = result2.x
        except:
            pass
            
    # Final check
    return best_coeffs

# EVOLVE-BLOCK-END
