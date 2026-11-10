# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv

# Constants
K = 10
M_VALUES = np.arange(-K, K + 1)

# Precompute normalization constants N_m for real orthonormal spherical harmonics
# Y_0 = N_0 P_0, Y_m = sqrt(2) N_m P_m cos(m phi) for m > 0,
#       sqrt(2) N_m P_|m| sin(|m| phi) for m < 0
# N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
N = np.zeros(2 * K + 1)
for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    # Using log-gamma for stability: log(N_m^2) = log(21/(4*pi)) + lgamma(11-abs_m) - lgamma(11+abs_m)
    from math import lgamma, log, pi
    log_n_sq = log(21.0 / (4.0 * pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
    N[i] = np.sqrt(np.exp(log_n_sq))

# Quadrature setup
# 70-point Gauss-Legendre in cos(theta) -> let x = cos(theta). x in [-1, 1].
# dx corresponds to d(cos theta). Integral over sphere is sum w_i * sum_j f(theta_i, phi_j) * (1/2) * (1/130) ?
# Wait, integral over S^2: int_0^pi int_0^{2pi} f(theta, phi) sin(theta) dphi dtheta
# Let x = cos(theta), dx = -sin(theta) dtheta.
# Integral = int_{-1}^1 int_0^{2pi} f(arccos(x), phi) dphi dx
# Gauss-Legendre approximates int_{-1}^1 g(x) dx = sum_{i=1}^n w_i g(x_i)
# Here g(x) = int_0^{2pi} f(arccos(x), phi) dphi.
# Approximate phi integral by 130 equally spaced points: (2pi/130) sum_{j=1}^{130} f(...)
# So total integral approx = sum_i w_i * (2pi/130) * sum_j f(x_i, phi_j)

n_theta = 70
n_phi = 130
x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

# x_gl is in [-1, 1]. theta = arccos(x).
# We need P_10^m(cos theta) = P_10^m(x_gl).
# Note: lpmv(m, l, x) gives P_l^m(x) with Condon-Shortley phase.
# x_gl shape (70,), phi_grid shape (130,)
# We want to compute Y_m(x, phi) for all m, x, phi.
# x_grid, phi_grid = np.meshgrid(x_gl, phi_grid, indexing='ij') -> shape (70, 130)
x_grid, phi_grid_2d = np.meshgrid(x_gl, phi_grid, indexing='ij')

# Precompute the basis functions Y_m(x, phi) for all m
# Shape: (21, 70, 130)
Y_basis = np.zeros((21, n_theta, n_phi))

for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    # Compute associated Legendre P_10^abs_m(x_gl)
    # lpmv expects m, l, x. m >= 0.
    # lpmv(abs_m, 10, x_gl)
    p_vals = lpmv(abs_m, 10, x_gl) # shape (70,)
    
    if m == 0:
        # Y_0 = N_0 P_0
        Y_basis[i] = N[i] * p_vals[:, np.newaxis]
    elif m > 0:
        # Y_m = sqrt(2) N_m P_m cos(m phi)
        cos_term = np.cos(m * phi_grid_2d) # shape (70, 130)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * cos_term
    else:
        # Y_m = sqrt(2) N_|m| P_|m| sin(|m| phi)
        sin_term = np.sin(abs_m * phi_grid_2d) # shape (70, 130)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * sin_term

# Precompute weights for integration
# Integral approx = sum_i w_gl[i] * (2pi/n_phi) * sum_j f[i,j]
# So we can define a weighted vector W of shape (70*130,) where W[i,j] = w_gl[i] * (2pi/n_phi)
W_flat = (w_gl[:, np.newaxis] * (2.0 * np.pi / n_phi)).flatten()

# Reshape Y_basis to (21, 70*130)
Y_flat = Y_basis.reshape(21, -1)

def evaluate_ratio(coeffs):
    """
    Computes R = ||f||_4 / ||f||_2 for given coefficients.
    coeffs: shape (21,)
    Returns: float ratio.
    """
    # f(x, phi) = sum_m c_m Y_m(x, phi)
    # f_flat = coeffs @ Y_flat -> shape (70*130,)
    f_flat = coeffs @ Y_flat
    
    # Norms with surface measure
    # ||f||_2^2 = int |f|^2 dOmega = sum W_flat * f_flat^2
    # ||f||_4^4 = int |f|^4 dOmega = sum W_flat * f_flat^4
    # Note: The problem says "norms over the unit sphere under surface measure".
    # ||f||_p = (int |f|^p dOmega)^{1/p}
    
    norm2_sq = np.sum(W_flat * f_flat**2)
    norm4_4 = np.sum(W_flat * f_flat**4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4)) # ( (|f|^4)^(1/4) )
    
    return norm4 / norm2

def refine_local(coeffs, time_left_fn):
    """
    Perform local optimization using Nelder-Mead or similar.
    We minimize R(coeffs).
    """
    from scipy.optimize import minimize
    
    def objective(c):
        return evaluate_ratio(c)
    
    # Use Nelder-Mead with more iterations and tighter tolerances
    # First pass: moderate settings
    result = minimize(objective, coeffs, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
    
    best_c = coeffs.copy()
    best_val = objective(coeffs)
    
    if result.fun < best_val:
        best_c = result.x.copy()
        best_val = result.fun
    
    # Second pass: high-precision polish from the best point
    if time_left_fn() > 2.0:
        result2 = minimize(objective, best_c, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        if result2.fun < best_val:
            best_c = result2.x.copy()
            best_val = result2.fun
    
    return best_c

def random_search(time_budget):
    best_coeffs = None
    best_val = np.inf
    start_time = time.time()
    
    # Helper to check time
    def time_left():
        return time_budget - (time.time() - start_time)
    
    # Initial candidates
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(21))
    
    # 2. Random Gaussian
    for _ in range(15):
        candidates.append(np.random.randn(21))
        
    # 3. Structured patterns
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) <= 2, 1, 0))
    candidates.append(np.ones(21) * np.where(M_VALUES % 2 == 0, 1, 0))
    candidates.append(np.ones(21) * np.where(M_VALUES % 2 != 0, 1, 0))
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) >= 8, 1, 0))
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) <= 5, 1, 0))
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) >= 5, 1, 0))
    
    # 4. Alternating signs patterns
    for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
        c = np.ones(21)
        for i, m in enumerate(M_VALUES):
            c[i] = np.cos(m * phase)
        candidates.append(c)
    
    # Evaluate initial candidates
    for c in candidates:
        val = evaluate_ratio(c)
        if val < best_val:
            best_val = val
            best_coeffs = c.copy()
            
    # Main search loop with adaptive step size
    step_scale = 0.3
    last_improvement_time = start_time
    
    while time_left() > 5.0:
        # Generate new candidates
        new_candidates = []
        
        if best_coeffs is not None:
            std_best = np.std(best_coeffs)
            if std_best < 1e-10:
                std_best = 1.0
            
            # 1. Multiple random walks with varying step sizes
            for scale_factor in [0.05, 0.1, 0.2, 0.5, 1.0]:
                noise = np.random.randn(21) * step_scale * scale_factor * std_best
                new_candidates.append(best_coeffs + noise)
                
            # 2. Sparse mutations
            for _ in range(3):
                c = best_coeffs.copy()
                n_mutate = np.random.randint(1, 5)
                indices = np.random.choice(21, n_mutate, replace=False)
                c[indices] += np.random.randn(n_mutate) * step_scale * std_best
                new_candidates.append(c)
            
            # 3. Random sparse vectors
            for _ in range(2):
                c = np.zeros(21)
                n_nonzero = np.random.randint(3, 8)
                indices = np.random.choice(21, n_nonzero, replace=False)
                c[indices] = np.random.choice([-1, 1], n_nonzero) * (0.5 + np.random.rand(n_nonzero))
                new_candidates.append(c)
        else:
            for _ in range(10):
                new_candidates.append(np.random.randn(21))
                
        # Evaluate new candidates
        for c in new_candidates:
            val = evaluate_ratio(c)
            if val < best_val:
                best_val = val
                best_coeffs = c.copy()
                last_improvement_time = time.time()
        
        # Adapt step size based on recent improvement
        if time.time() - last_improvement_time > 10.0:
            step_scale *= 0.5  # Reduce step if no recent improvement
        else:
            step_scale = min(step_scale * 1.1, 1.0)  # Gradually increase if improving
        
        # If we have a good candidate, try local refinement
        if best_coeffs is not None and time_left() > 15.0:
            refined = refine_local(best_coeffs, time_left)
            refined_val = evaluate_ratio(refined)
            if refined_val < best_val:
                best_val = refined_val
                best_coeffs = refined.copy()
                last_improvement_time = time.time()
                
    return best_coeffs

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    # Time budget management
    start_time = time.time()
    max_time = 580.0 # Safety margin from 630s (leave 50s buffer)
    
    # Run search
    try:
        best_coeffs = random_search(max_time)
    except Exception as e:
        # Fallback to baseline if something goes wrong
        best_coeffs = np.ones(21)
        
    # Final validation and cleanup
    if best_coeffs is None or not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
        
    # Ensure size is 21
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
        
    # Check max abs coefficient constraints
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        # Normalize to be within range, e.g., scale to max 1.0
        if max_c > 0:
            best_coeffs = best_coeffs / max_c * 1.0
        else:
            best_coeffs = np.ones(21)
    
    # Final high-precision polish if time permits
    elapsed = time.time() - start_time
    if elapsed < max_time - 30 and best_coeffs is not None and np.all(np.isfinite(best_coeffs)):
        try:
            refined = refine_local(best_coeffs, lambda: max_time - (time.time() - start_time))
            if np.all(np.isfinite(refined)):
                val_orig = evaluate_ratio(best_coeffs)
                val_ref = evaluate_ratio(refined)
                if val_ref < val_orig:
                    best_coeffs = refined
        except:
            pass
    
    # Final validation after polish
    if best_coeffs is None or not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        if max_c > 0:
            best_coeffs = best_coeffs / max_c * 1.0
        else:
            best_coeffs = np.ones(21)
            
    return best_coeffs

# EVOLVE-BLOCK-END
