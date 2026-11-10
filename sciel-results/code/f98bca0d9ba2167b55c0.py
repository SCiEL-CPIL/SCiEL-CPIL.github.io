# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

# Constants
K = 10
M_VALUES = np.arange(-K, K + 1)

# Precompute normalization constants N_m for real orthonormal spherical harmonics
N = np.zeros(2 * K + 1)
for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    from math import lgamma, log, pi
    log_n_sq = log(21.0 / (4.0 * pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
    N[i] = np.sqrt(np.exp(log_n_sq))

# Quadrature setup
n_theta = 70
n_phi = 130
x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

x_grid, phi_grid_2d = np.meshgrid(x_gl, phi_grid, indexing='ij')

# Precompute the basis functions Y_m(x, phi) for all m
Y_basis = np.zeros((21, n_theta, n_phi))

for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    p_vals = lpmv(abs_m, 10, x_gl)
    
    if m == 0:
        Y_basis[i] = N[i] * p_vals[:, np.newaxis]
    elif m > 0:
        cos_term = np.cos(m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * cos_term
    else:
        sin_term = np.sin(abs_m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * sin_term

# Precompute weights for integration
W_flat = (w_gl[:, np.newaxis] * (2.0 * np.pi / n_phi)).flatten()
Y_flat = Y_basis.reshape(21, -1)

def evaluate_ratio_full(coeffs):
    """
    Computes R = ||f||_4 / ||f||_2 for given full 21-coeff vector.
    """
    f_flat = coeffs @ Y_flat
    norm2_sq = np.sum(W_flat * f_flat**2)
    norm4_4 = np.sum(W_flat * f_flat**4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2

def evaluate_ratio_symmetric(s):
    """
    Computes R for a symmetric combination defined by squared amplitudes s (length 11).
    s[0] corresponds to m=0. s[m] corresponds to the pair m, -m.
    We assume c_m = sqrt(s[m]) * sign. The sign doesn't matter for L4 if we allow
    arbitrary phases, but here we fix a specific phase structure to reduce variables.
    
    Actually, to be rigorous:
    f = c_0 Y_0 + sum_{m=1}^{10} (c_m Y_m + c_{-m} Y_{-m})
    Let c_m = A_m cos(alpha_m), c_{-m} = A_m sin(alpha_m).
    Then c_m^2 + c_{-m}^2 = A_m^2.
    The term in f is A_m (cos(alpha_m) Y_m + sin(alpha_m) Y_{-m}).
    Note Y_m = sqrt(2) N_m P^m cos(m phi), Y_{-m} = sqrt(2) N_m P^m sin(m phi).
    So the term is sqrt(2) N_m P^m A_m (cos(alpha_m) cos(m phi) + sin(alpha_m) sin(m phi))
    = sqrt(2) N_m P^m A_m cos(m phi - alpha_m).
    
    This shows that for any fixed magnitudes A_m (and c_0), we can rotate the phase
    alpha_m to minimize the L4 norm (or maximize L2, but L2 is fixed by orthogonality).
    Wait, L2 is fixed by the sum of squares of coefficients due to orthonormality.
    ||f||_2^2 = c_0^2 + sum (c_m^2 + c_{-m}^2) = c_0^2 + sum A_m^2.
    So L2 is constant for fixed s = [c_0^2, A_1^2, ..., A_10^2].
    
    To minimize L4, we need to minimize sum w_i (f(x_i))^4.
    f(x, phi) = c_0 Y_0(x) + sum_{m=1}^{10} sqrt(2) N_m P^m(x) A_m cos(m phi - alpha_m).
    
    The cross terms in f^4 depend on the phases alpha_m.
    Minimizing over alpha_m is a separate problem.
    However, for a "flattest" harmonic, we might expect certain symmetries.
    
    Let's simplify the strategy:
    Instead of optimizing over 21 variables, we optimize over 11 variables s = [c_0^2, A_1^2, ...]
    AND we assume the "best" phase alignment is the one that makes the function as "isotropic"
    as possible or simply use a heuristic phase.
    
    A common heuristic for flattening is to align the peaks of different m components
    such that they don't reinforce each other too strongly.
    However, a more robust approach is to note that for a given set of magnitudes,
    the minimum L4 norm over phases is a difficult non-convex problem.
    
    Alternative Structural Change:
    Use the fact that the L4 norm is minimized when the function is as close to constant
    as possible.
    
    Let's try a different optimization variable set:
    Optimize over the 11 magnitudes s, and for each s, perform a quick 1D or 2D minimization
    over the critical phases if necessary, OR assume a specific phase pattern that is
    known to be good (e.g., all phases 0, or phases that cancel).
    
    Given the time budget, let's assume the optimal solution has a specific symmetry:
    c_{-m} = c_m (real coefficients, even symmetry in phi) OR c_{-m} = -c_m (odd).
    Actually, the "flattest" function is likely to be real and have specific parity.
    
    Let's stick to the 21 variables but use L-BFGS-B with analytic gradient.
    The gradient of R = N4/N2 w.r.t c is:
    dR/dc = (1/N2) dN4/dc - (N4/N2^2) dN2/dc
    dN2/dc = 2 N2 * (N2_sq / N2^2)? No.
    N2 = sqrt( c^T G c ) where G is the Gram matrix (identity for orthonormal basis on sphere,
    but discrete quadrature might have slight errors, let's compute G_discrete).
    Actually, the basis is orthonormal under exact integration. Under quadrature, it's nearly so.
    Let's compute G = Y_flat^T (W_flat * Y_flat) ? No, G_ij = sum w_k Y_i(x_k) Y_j(x_k).
    
    dN4/dc:
    N4^4 = sum w_k (f(x_k))^4 = sum w_k (sum c_j Y_j(x_k))^4.
    This is a 4th order polynomial.
    d/dc_i (sum w_k f_k^4) = sum w_k 4 f_k^3 (sum_j Y_jk c_j ... wait, f_k = sum_j c_j Y_jk)
    = sum w_k 4 f_k^3 Y_ik.
    
    So grad_N4_sq = 4 * (Y_flat * (W_flat * f_flat**3)) @ ?
    Let v = W_flat * f_flat**3 (vector of length n_points)
    grad_N4_sq_i = 4 * sum_k v_k Y_ik = 4 * (Y_flat.T @ v)_i
    
    dN2_sq/dc_i = 2 * (Y_flat.T @ (W_flat * f_flat))_i
    
    Let N2_sq = c^T G c (approx identity)
    N4_sq = sum w f^4
    
    R = (N4_sq^(1/4)) / (N2_sq^(1/2))
    ln R = 0.25 ln N4_sq - 0.5 ln N2_sq
    d ln R / dc = 0.25 (1/N4_sq) dN4_sq/dc - 0.5 (1/N2_sq) dN2_sq/dc
    
    grad_R = R * (0.25 * grad_N4_sq / N4_sq - 0.5 * grad_N2_sq / N2_sq)
    
    This is computable in O(N_points * N_basis).
    """
    pass

def compute_grad_R(c):
    """
    Compute gradient of R(c) = ||f||_4 / ||f||_2.
    """
    f_flat = c @ Y_flat
    
    norm2_sq = np.sum(W_flat * f_flat**2)
    norm4_sq = np.sum(W_flat * f_flat**4)
    
    if norm2_sq <= 0 or norm4_sq <= 0:
        return np.zeros(21)
    
    # dN2_sq/dc = 2 * Y_flat.T @ (W_flat * f_flat)
    w_f = W_flat * f_flat
    grad_N2_sq = 2.0 * (Y_flat.T @ w_f)
    
    # dN4_sq/dc = 4 * Y_flat.T @ (W_flat * f_flat**3)
    w_f3 = W_flat * (f_flat**3)
    grad_N4_sq = 4.0 * (Y_flat.T @ w_f3)
    
    R = np.sqrt(np.sqrt(norm4_sq)) / np.sqrt(norm2_sq)
    
    # grad_R = R * (0.25 * grad_N4_sq / norm4_sq - 0.5 * grad_N2_sq / norm2_sq)
    grad_R = R * (0.25 * grad_N4_sq / norm4_sq - 0.5 * grad_N2_sq / norm2_sq)
    
    return grad_R

def objective_and_grad(c):
    """
    Objective: R(c)
    Gradient: grad_R(c)
    """
    f_flat = c @ Y_flat
    
    norm2_sq = np.sum(W_flat * f_flat**2)
    norm4_sq = np.sum(W_flat * f_flat**4)
    
    if norm2_sq <= 0:
        return 1e10, np.zeros(21)
    
    R = np.sqrt(np.sqrt(norm4_sq)) / np.sqrt(norm2_sq)
    
    w_f = W_flat * f_flat
    grad_N2_sq = 2.0 * (Y_flat.T @ w_f)
    
    w_f3 = W_flat * (f_flat**3)
    grad_N4_sq = 4.0 * (Y_flat.T @ w_f3)
    
    grad_R = R * (0.25 * grad_N4_sq / norm4_sq - 0.5 * grad_N2_sq / norm2_sq)
    
    return R, grad_R

def run_search(time_budget):
    start_time = time.time()
    def time_left():
        return time_budget - (time.time() - start_time)
    
    best_val = np.inf
    best_coeffs = np.ones(21)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Zonal
    c_zonal = np.zeros(21)
    c_zonal[10] = 1.0
    starts.append(c_zonal)
    
    # 2. Pure m=10
    c_pure = np.zeros(21)
    c_pure[20] = 1.0
    starts.append(c_pure)
    
    # 3. Random starts
    np.random.seed(42)
    for _ in range(100):
        c = np.random.randn(21)
        # Normalize to unit L2 to start on the unit sphere of coefficients
        # (R is scale invariant, so we can fix norm)
        norm = np.linalg.norm(c)
        if norm > 1e-6:
            c = c / norm
        starts.append(c)
        
        # Symmetric starts
        c_sym = np.zeros(21)
        c_sym[10] = np.random.randn()
        for m in range(1, 11):
            val = np.random.randn()
            c_sym[10 + m] = val
            c_sym[10 - m] = val # or -val
        norm = np.linalg.norm(c_sym)
        if norm > 1e-6:
            c_sym = c_sym / norm
        starts.append(c_sym)
        
        # Anti-symmetric
        c_anti = np.zeros(21)
        # c_anti[10] = 0
        for m in range(1, 11):
            val = np.random.randn()
            c_anti[10 + m] = val
            c_anti[10 - m] = -val
        norm = np.linalg.norm(c_anti)
        if norm > 1e-6:
            c_anti = c_anti / norm
        starts.append(c_anti)

    np.random.shuffle(starts)
    
    # Optimize each start using L-BFGS-B
    for i, c_init in enumerate(starts):
        if time_left() < 1.0:
            break
        
        def obj(c):
            val, _ = objective_and_grad(c)
            return val
        
        def grad(c):
            _, g = objective_and_grad(c)
            return g
        
        try:
            # L-BFGS-B is efficient for smooth functions with gradient
            # We don't need to constrain norm because R is scale invariant,
            # but it helps to keep coefficients in a reasonable range to avoid
            # numerical issues.
            # We can add a penalty or just let it run.
            # Since R is scale invariant, the gradient will have a component
            # that drives the norm to a specific value (where Hessian is well-behaved).
            
            res = minimize(obj, c_init, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
                
        except Exception:
            pass

    # Final polish on best solution with higher precision
    if time_left() > 2.0:
        try:
            res = minimize(obj, best_coeffs, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

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
    max_time = 600.0 # Safety margin from 630s limit
    
    try:
        best_coeffs = run_search(max_time)
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
            
    return best_coeffs

# EVOLVE-BLOCK-END
