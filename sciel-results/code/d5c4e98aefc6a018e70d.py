# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time
import math

def _precompute_gaunt():
    """
    Precompute the Gaunt integral tensor for degree-10 real spherical harmonics.
    
    We need: ∫ Y_a Y_b Y_c dΩ for all a,b,c in {-10,...,10}
    
    Using the identity for real spherical harmonics:
    ∫ Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3} dΩ = sqrt((2l1+1)(2l2+1)(2l3+1)/(4π))
        * (l1 l2 l3; 0 0 0) * (l1 l2 l3; m1 m2 m3)
    
    where the parentheticals are Wigner 3j symbols.
    
    For our case, all l = 10, so we need:
    ∫ Y_{10}^{m1} Y_{10}^{m2} Y_{10}^{m3} dΩ = sqrt(21^3/(4π)) * (10 10 10; 0 0 0) * (10 10 10; m1 m2 m3)
    
    The coefficient (10 10 10; 0 0 0) is a constant.
    """
    from scipy.special import lpmv
    import scipy.integrate as integrate
    
    # We'll compute the Gaunt integrals numerically using quadrature
    # since it's simpler and we need it for the specific case l1=l2=l3=10
    
    # Gauss-Legendre quadrature in cos(theta)
    n_theta = 70
    mu, w_mu = np.polynomial.legendre.leggauss(n_theta)
    
    # Equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    dphi = 2*np.pi / n_phi
    
    # Build grid
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing='ij')
    theta_grid = np.arccos(mu_grid)
    
    # Weights
    w_grid = np.outer(w_mu, np.full(n_phi, dphi))
    
    # Precompute basis functions Y_m for m = -10..10
    # Y_0 = N_0 * P_10^0(cos theta)
    # Y_m = sqrt(2) * N_m * P_10^|m|(cos theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^|m|(cos theta) * sin(|m|*phi) for m < 0
    
    k = 10
    n_m = 2*k + 1  # 21
    
    # Normalization constants
    N = np.zeros(n_m)
    for m in range(n_m):
        m_val = m - k
        abs_m = abs(m_val)
        N[m] = np.sqrt((2*k + 1)/(4*np.pi) * math.factorial(k - abs_m)/math.factorial(k + abs_m))
    
    # Associated Legendre polynomials with Condon-Shortley phase
    # scipy.special.lpmv(m, l, x) includes the Condon-Shortley phase (-1)^m
    P = np.zeros((n_m, n_theta))
    for m in range(n_m):
        m_val = m - k
        abs_m = abs(m_val)
        P[m, :] = lpmv(abs_m, k, mu)
    
    # Build basis functions
    Y = np.zeros((n_m, n_theta, n_phi))
    for m in range(n_m):
        m_val = m - k
        if m_val == 0:
            Y[m, :, :] = N[m] * P[m, :][:, np.newaxis] * np.ones((1, n_phi))
        elif m_val > 0:
            Y[m, :, :] = np.sqrt(2) * N[m] * P[m, :][:, np.newaxis] * np.cos(m_val * phi_grid)
        else:
            abs_m = -m_val
            Y[m, :, :] = np.sqrt(2) * N[m] * P[m, :][:, np.newaxis] * np.sin(abs_m * phi_grid)
    
    # Compute Gaunt integrals: G[a,b,c] = ∫ Y_a Y_b Y_c dΩ
    # This is a 21x21x21 tensor
    G = np.zeros((n_m, n_m, n_m))
    
    # To save time, we can compute this more efficiently
    # G[a,b,c] = sum_{i,j} w[i,j] Y[a,i,j] Y[b,i,j] Y[c,i,j]
    
    # Flatten for efficiency
    Y_flat = Y.reshape(n_m, -1)  # (21, 70*130)
    w_flat = w_grid.flatten()
    
    # Compute using matrix operations
    # G[a,b,c] = Y_flat[a] @ diag(w_flat) @ Y_flat[b] * ... 
    # Actually, we need the triple product
    # Let's do it in a loop over one index for memory efficiency
    
    # Alternative: compute the 4th moment tensor
    # M[a,b,c,d] = ∫ Y_a Y_b Y_c Y_d dΩ
    # Then ||f||_4^4 = sum_{a,b,c,d} c_a c_b c_c c_d M[a,b,c,d]
    # And ||f||_2^2 = sum_{a,b} c_a c_b delta_{a,b} (by orthonormality)
    
    # Actually, let's compute the 4th moment directly
    # M[a,b,c,d] = sum_{i,j} w[i,j] Y[a,i,j] Y[b,i,j] Y[c,i,j] Y[d,i,j]
    
    # This is a 21^4 tensor which is 194481 elements - manageable
    # But computing it naively is expensive
    
    # Better approach: compute ||f||_4^4 and ||f||_2^2 directly for given coefficients
    # and use numerical differentiation or automatic differentiation for gradients
    
    return Y_flat, w_flat, N, P, mu, phi, n_theta, n_phi

def _compute_R(coeffs, Y_flat, w_flat):
    """
    Compute R = ||f||_4 / ||f||_2 for given coefficients.
    
    coeffs: array of shape (21,)
    Y_flat: precomputed basis functions, shape (21, n_points)
    w_flat: quadrature weights, shape (n_points,)
    """
    # f = sum_m c_m Y_m
    f = Y_flat.T @ coeffs  # shape (n_points,)
    
    # ||f||_2^2 = sum w_i f_i^2
    norm2_sq = np.sum(w_flat * f**2)
    
    # ||f||_4^4 = sum w_i f_i^4
    norm4_4 = np.sum(w_flat * f**4)
    
    if norm2_sq <= 0:
        return np.inf
    
    R = (norm4_4 / norm2_sq**2)**0.25
    return R

def _compute_grad_R(coeffs, Y_flat, w_flat):
    """
    Compute gradient of R with respect to coeffs.
    
    R = ||f||_4 / ||f||_2 = (sum w_i f_i^4)^(1/4) / (sum w_i f_i^2)^(1/2)
    
    Let A = ||f||_4^4 = sum w_i f_i^4
    Let B = ||f||_2^2 = sum w_i f_i^2
    
    R = A^(1/4) / B^(1/2)
    
    dR/da = (1/4) A^(-3/4) B^(-1/2) * dA/da - (1/2) A^(1/4) B^(-3/2) * dB/da
    
    dA/da = sum_i w_i * 4 * f_i^3 * df_i/da = 4 * sum_i w_i f_i^3 Y_a[i]
    dB/da = sum_i w_i * 2 * f_i * df_i/da = 2 * sum_i w_i f_i Y_a[i]
    
    So dR/da = (1/4) A^(-3/4) B^(-1/2) * 4 * sum_i w_i f_i^3 Y_a[i]
              - (1/2) A^(1/4) B^(-3/2) * 2 * sum_i w_i f_i Y_a[i]
    
    Simplifying:
    dR/da = A^(-3/4) B^(-1/2) * sum_i w_i f_i^3 Y_a[i]
            - A^(1/4) B^(-3/2) * sum_i w_i f_i Y_a[i]
    
    = (1/(A^(3/4) B^(1/2))) * sum_i w_i f_i^3 Y_a[i]
      - (1/(A^(1/4) B^(3/2))) * sum_i w_i f_i Y_a[i]
    
    Let alpha = 1/(A^(3/4) B^(1/2))
    Let beta = 1/(A^(1/4) B^(3/2))
    
    dR/da = alpha * (w * f^3) @ Y[a] - beta * (w * f) @ Y[a]
    """
    f = Y_flat.T @ coeffs
    A = np.sum(w_flat * f**4)
    B = np.sum(w_flat * f**2)
    
    if A <= 0 or B <= 0:
        return np.zeros_like(coeffs)
    
    alpha = 1.0 / (A**0.75 * B**0.5)
    beta = 1.0 / (A**0.25 * B**1.5)
    
    # grad_a = alpha * sum_i w_i f_i^3 Y[a,i] - beta * sum_i w_i f_i Y[a,i]
    # = alpha * (w * f^3) @ Y[a] - beta * (w * f) @ Y[a]
    
    wf3 = w_flat * f**3
    wf = w_flat * f
    
    grad = alpha * (Y_flat @ wf3) - beta * (Y_flat @ wf)
    
    return grad

def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    that MINIMIZE the norm ratio ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}.
    """
    start_time = time.time()
    deadline = start_time + 620  # Leave 10s safety margin
    
    k = 10
    n_m = 2*k + 1  # 21
    
    # Precompute basis
    Y_flat, w_flat, N, P, mu, phi, n_theta, n_phi = _precompute_gaunt()
    
    # Try multiple initial points and optimize each
    # Use L-BFGS-B from scipy.optimize
    
    from scipy.optimize import minimize
    
    # Define objective and gradient
    def objective(c):
        return _compute_R(c, Y_flat, w_flat)
    
    def gradient(c):
        return _compute_grad_R(c, Y_flat, w_flat)
    
    # Generate diverse initial points
    rng = np.random.default_rng(42)
    
    initial_points = []
    
    # 1. All ones
    initial_points.append(np.ones(n_m))
    
    # 2. More random points for diversity
    for _ in range(30):
        c = rng.normal(0, 1, n_m)
        initial_points.append(c)
    
    # 3. Structured patterns
    # Anti-symmetric: c[m] = -c[2k-m]
    for _ in range(3):
        c = np.zeros(n_m)
        for m in range(k):
            c[m] = rng.normal(0, 1)
            c[2*k - m] = -c[m]
        initial_points.append(c)
    
    # Paired: c[m] = c[2k-m]
    for _ in range(3):
        c = np.zeros(n_m)
        for m in range(k):
            c[m] = rng.normal(0, 1)
            c[2*k - m] = c[m]
        initial_points.append(c)
    
    # Comb pattern: alternating signs
    c = np.ones(n_m) * np.array([1 if (m % 2 == 0) else -1 for m in range(n_m)])
    initial_points.append(c)
    
    # Sine pattern
    m_vals = np.arange(-k, k+1)
    c = np.cos(np.pi * m_vals / k)
    initial_points.append(c)
    
    # Cosine pattern
    c = np.sin(np.pi * m_vals / k)
    initial_points.append(c)
    
    # Gaussian decay
    c = np.exp(-0.5 * (m_vals / 5)**2)
    initial_points.append(c)
    
    # Polynomial in m
    c = 1 + 0.1*m_vals + 0.01*m_vals**2
    initial_points.append(c)
    
    # Sparse patterns: only even m or only odd m
    c_even = np.zeros(n_m)
    for m in range(0, n_m, 2):
        c_even[m] = rng.normal(0, 1)
    initial_points.append(c_even)
    
    c_odd = np.zeros(n_m)
    for m in range(1, n_m, 2):
        c_odd[m] = rng.normal(0, 1)
    initial_points.append(c_odd)
    
    # Low-order emphasis: weight by exp(-|m|/3)
    c = np.exp(-np.abs(m_vals) / 3.0) * rng.normal(1, 0.5, n_m)
    initial_points.append(c)
    
    # High-order emphasis: weight by exp(-|m|/8)
    c = np.exp(-np.abs(m_vals) / 8.0) * rng.normal(1, 0.5, n_m)
    initial_points.append(c)
    
    best_R = np.inf
    best_coeffs = None
    
    for i, c0 in enumerate(initial_points):
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(
                objective,
                c0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if result.success or result.fun < best_R:
                R_val = result.fun
                if np.isfinite(R_val) and R_val < best_R:
                    best_R = R_val
                    best_coeffs = result.x.copy()
        except Exception as e:
            pass
    
    # If no valid result found, fall back to all ones
    if best_coeffs is None:
        best_coeffs = np.ones(n_m)
        best_R = _compute_R(best_coeffs, Y_flat, w_flat)
    
    # Local restarts with decaying noise around the best solution
    # This helps escape shallow local minima
    if time.time() < deadline - 30:
        n_restarts = 8
        noise_scale = 0.1
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Add decaying noise
            noise = noise_scale * np.exp(-0.5 * restart / 3.0) * rng.normal(0, 1, n_m)
            c_perturbed = best_coeffs + noise
            
            try:
                result = minimize(
                    objective,
                    c_perturbed,
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-11, 'maxfun': 7000}
                )
                
                if np.isfinite(result.fun) and result.fun < best_R:
                    best_R = result.fun
                    best_coeffs = result.x.copy()
            except Exception:
                pass
    
    # High-precision polish with tighter tolerances
    if time.time() < deadline - 10:
        try:
            result = minimize(
                objective,
                best_coeffs,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 50000}
            )
            if np.isfinite(result.fun) and result.fun < best_R:
                best_R = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Ultra-fine polish if time permits
    if time.time() < deadline - 5:
        try:
            result = minimize(
                objective,
                best_coeffs,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 100000}
            )
            if np.isfinite(result.fun) and result.fun < best_R:
                best_R = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Validate
    if best_coeffs is None or len(best_coeffs) != 21:
        return np.ones(21)
    
    if np.any(~np.isfinite(best_coeffs)):
        return np.ones(21)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Scale to valid range
        if max_abs < 1e-3:
            best_coeffs = best_coeffs / max_abs * 1.0
        elif max_abs > 1e5:
            best_coeffs = best_coeffs / max_abs * 1.0
    
    return best_coeffs

# EVOLVE-BLOCK-END
