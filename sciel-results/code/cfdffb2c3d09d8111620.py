# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

# Constants for the problem
K = 10
N_POINTS = 2 * K + 1  # 21 coefficients

# Precomputation of quadrature and basis functions
# 70-point Gauss-Legendre in cos(theta)
# 130 equally spaced phi
# Total points = 70 * 130 = 9100

def _precompute_basis():
    """
    Precompute the spherical harmonics basis matrix Y and weights W.
    Y is shape (N_points, 21).
    W is shape (N_points,).
    """
    # Gauss-Legendre nodes and weights
    # x is cos(theta), w is weight
    x, w = np.polynomial.legendre.leggauss(70)
    
    # phi points
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    
    # Meshgrid
    X, Phi = np.meshgrid(x, phi, indexing='ij')
    
    # Flatten to 1D arrays of points
    cos_theta = X.flatten()
    phi_flat = Phi.flatten()
    
    # Weights
    # Surface element dOmega = dphi * d(cos_theta)
    # The integral is sum_i w_i * (2*pi/130) * f(x_i, phi_j)
    # We want the norm squared to be sum w_i * f(x_i)^2
    # So the effective weight for each point is w_i * (2*pi/130)
    weights = np.repeat(w, 130) * (2 * np.pi / 130)
    
    # Compute Spherical Harmonics
    # Y_m for m = -10..10
    # Real orthonormal harmonics:
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    
    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    
    Y = np.zeros((len(cos_theta), 21))
    
    for m_idx in range(21):
        m = m_idx - 10
        abs_m = abs(m)
        
        # Normalization factor
        # Using log factorial for stability, though for k=10 it's fine
        n_m = np.sqrt(21.0 / (4.0 * np.pi) * factorial(10 - abs_m) / factorial(10 + abs_m))
        
        # Associated Legendre Polynomial
        # lpmv(m, l, x) includes Condon-Shortley phase (-1)^m
        P_m = lpmv(abs_m, 10, cos_theta)
        
        if m == 0:
            Y[:, m_idx] = n_m * P_m
        elif m > 0:
            Y[:, m_idx] = np.sqrt(2.0) * n_m * P_m * np.cos(m * phi_flat)
        else:
            Y[:, m_idx] = np.sqrt(2.0) * n_m * P_m * np.sin(abs_m * phi_flat)
            
    return Y, weights

# Precompute at module load time to save time during optimization
Y_BASIS, W_QUAD = _precompute_basis()

# Precompute the 4th order tensor contraction for the L4 norm
# ||f||_4^4 = sum_i W_i * (sum_j Y_ij c_j)^4
# Let A_ij = W_i^{1/4} Y_ij. Then ||f||_4 = (sum_i (sum_j A_ij c_j)^4)^{1/4}
# This is not a simple quadratic form in c, it's a quartic form.
# However, we can compute the gradient analytically.
# f_i = Y c
# J = sum W f^4
# dJ/dc_j = 4 sum W f^3 Y_ij

def _objective_and_gradient(c):
    """
    Compute the L4/L2 ratio and its gradient.
    Since the basis is orthonormal, ||c||_2 = ||f||_2 (approximately, via quadrature).
    Actually, the problem asks for ||f||_4 / ||f||_2.
    With the given quadrature, the orthonormality is exact for degree 10.
    So ||f||_2^2 = c^T c.
    We want to minimize R = ||f||_4 / sqrt(c^T c).
    Equivalent to minimizing ||f||_4^4 / (c^T c)^2.
    Let's minimize log(R) or just R.
    Let's minimize F(c) = ||f||_4^4 / (c^T c)^2.
    """
    c = np.asarray(c)
    f = Y_BASIS @ c
    
    norm2_sq = np.dot(c, c)
    if norm2_sq < 1e-12:
        return 1e12, np.zeros_like(c)
    
    norm4_pow4 = np.sum(W_QUAD * f**4)
    
    # R = (norm4_pow4 / (4*pi)) ** 0.25 / sqrt(norm2_sq)
    # Wait, the norm is defined over the sphere with surface measure.
    # The quadrature weights sum to 4*pi.
    # ||f||_2^2 = sum W f^2. Since Y is orthonormal w.r.t this measure, sum W Y_i Y_j = delta_ij.
    # So sum W f^2 = c^T c.
    # ||f||_4 = (sum W f^4)^(1/4).
    # Ratio R = (sum W f^4)^(1/4) / sqrt(c^T c).
    
    # We want to minimize R.
    # Let's minimize R^4 to avoid roots: G(c) = (sum W f^4) / (c^T c)^2
    
    val = norm4_pow4 / (norm2_sq ** 2)
    
    # Gradient of G
    # G = N / D, N = sum W f^4, D = (c^T c)^2
    # dN/dc = 4 Y^T (W f^3)
    # dD/dc = 4 (c^T c) c
    # dG/dc = (dN D - N dD) / D^2
    #       = (4 Y^T (W f^3) (c^T c)^2 - N 4 (c^T c) c) / (c^T c)^4
    #       = 4 / (c^T c)^2 * (Y^T (W f^3) (c^T c) - N c)
    #       = 4 / norm2_sq * (Y^T (W f^3) - (N / norm2_sq) c)
    
    grad_N = 4.0 * (Y_BASIS.T @ (W_QUAD * f**3))
    grad_G = (4.0 / norm2_sq) * (grad_N - (norm4_pow4 / norm2_sq) * c)
    
    return val, grad_G

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c minimizing L4/L2 ratio for degree 10 spherical harmonics.
    """
    start_time = time.time()
    deadline = 600.0 # Safety margin
    
    best_c = None
    best_val = float('inf')
    
    # Multi-start optimization
    # The problem is scale-invariant, so we can fix ||c||_2 = 1 or just let the optimizer handle it.
    # L-BFGS-B works well with analytic gradients.
    
    # Initial guesses
    # 1. All ones
    # 2. Random
    # 3. Single mode (m=0, m=10, etc.)
    
    inits = []
    inits.append(np.ones(N_POINTS))
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        v = rng.normal(size=N_POINTS)
        inits.append(v)
        
    # Single modes
    for i in range(N_POINTS):
        v = np.zeros(N_POINTS)
        v[i] = 1.0
        inits.append(v)
        
    # Low m weighted (often flatter)
    v = np.zeros(N_POINTS)
    for i in range(N_POINTS):
        m = abs(i - 10)
        v[i] = 1.0 / (1.0 + m)
    inits.append(v)
    
    # High m weighted
    v = np.zeros(N_POINTS)
    for i in range(N_POINTS):
        m = abs(i - 10)
        v[i] = 1.0 + m
    inits.append(v)
    
    for c0 in inits:
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(
                _objective_and_gradient,
                c0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_val:
                # Check validity
                c_res = res.x
                if np.all(np.isfinite(c_res)) and np.max(np.abs(c_res)) >= 1e-3 and np.max(np.abs(c_res)) <= 1e5:
                    # Normalize to avoid issues with extreme scales, though ratio is invariant
                    # The problem allows any scale.
                    best_val = res.fun
                    best_c = c_res
        except Exception:
            pass
            
    if best_c is None:
        # Fallback to all ones if optimization failed completely
        return np.ones(N_POINTS)
        
    return best_c

# EVOLVE-BLOCK-END
