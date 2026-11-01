"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute Gauss-Hermite quadrature nodes and weights
# For physicists' Hermite: ∫ f(x) e^{-x²} dx ≈ sum_i w_i f(x_i)
GH_N = 50
xgh, wgh = np.polynomial.hermite.hermgauss(GH_N)

# Precompute Hermite polynomials at quadrature nodes for degrees 0, 4, 8, 12
def hermite_poly(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    H0 = np.ones_like(x)
    H1 = 2.0 * x
    for k in range(1, n):
        H2 = 2.0 * x * H1 - 2.0 * k * H0
        H0 = H1
        H1 = H2
    return H1

# Precompute H_n at quadrature points
H0_nodes = hermite_poly(0, xgh)
H4_nodes = hermite_poly(4, xgh)
H8_nodes = hermite_poly(8, xgh)
H12_nodes = hermite_poly(12, xgh)

# H_n(0) values for the constraint P(0) = 0
H0_0 = 1.0
H4_0 = -3.0
H8_0 = 105.0
H12_0 = -10395.0

# Weights for integrals: w_i * e^{-x_i^2} is already in wgh for hermgauss
# Actually hermgauss gives weights for ∫ f(x) e^{-x²} dx, so we just use wgh directly.

def compute_c12(c0, c4, c8):
    """Compute c12 such that P(0) = 0."""
    return (c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / (-H12_0)

def objective(coeffs):
    """Compute the uncertainty product bound."""
    c0, c4, c8 = coeffs
    c12 = compute_c12(c0, c4, c8)
    
    # P(x) at quadrature nodes
    P = c0 * H0_nodes + c4 * H4_nodes + c8 * H8_nodes + c12 * H12_nodes
    
    # ||g||^2 = ∫ P^2 e^{-x²} dx
    norm_sq = np.sum(wgh * P * P)
    
    if norm_sq <= 1e-15:
        return 1e10
    
    # σx² = ∫ x² P² e^{-x²} dx / norm_sq
    x2_P2 = np.sum(wgh * xgh**2 * P * P)
    sigma_x2 = x2_P2 / norm_sq
    
    # g'(x) = e^{-x²/2} (P'(x) - x*P(x))
    # Need P'(x): H0'=0, H4'=8x³-12x, H8'=16x⁷-224x⁵+672x³-336x, H12' = 24*H11
    # Let's compute P'(x) using derivatives of Hermite polynomials
    # H_n'(x) = 2n H_{n-1}(x)
    H0_prime = np.zeros_like(xgh)
    H4_prime = 8.0 * hermite_poly(3, xgh)
    H8_prime = 16.0 * hermite_poly(7, xgh)
    H12_prime = 24.0 * hermite_poly(11, xgh)
    
    P_prime = c0 * H0_prime + c4 * H4_prime + c8 * H8_prime + c12 * H12_prime
    
    Q = P_prime - xgh * P
    momentum_sq = np.sum(wgh * Q * Q)
    sigma_p2 = momentum_sq / norm_sq
    
    # Uncertainty product: σx * σp
    # The bound on C4 relates to this product
    return np.sqrt(sigma_x2 * sigma_p2)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # 170s safety margin
    
    # Multiple starting points
    starting_points = [
        [-1.13, 0.04, 0.000308],   # Seed
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [-2.0, 0.1, 0.01],
        [-0.5, 0.02, 0.001],
        [-1.5, 0.05, 0.0005],
        [0.5, -0.03, 0.002],
        [-0.8, 0.01, -0.001],
    ]
    
    best_val = 1e10
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    # Evaluate seed first
    seed_val = objective(np.array([-113/100, 1/25, 1/3240]))
    if seed_val < best_val:
        best_val = seed_val
    
    for sp in starting_points:
        if time.time() > deadline:
            break
        sp_arr = np.array(sp, dtype=float)
        try:
            result = minimize(
                objective, sp_arr, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Try Nelder-Mead for a few good starting points
    if time.time() < deadline - 30:
        for sp in starting_points[:6]:
            if time.time() > deadline:
                break
            sp_arr = np.array(sp, dtype=float)
            try:
                result = minimize(
                    objective, sp_arr, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_coeffs = result.x
            except Exception:
                pass
    
    # Coarse grid search if time permits
    if time.time() < deadline - 60:
        c0_range = np.linspace(-3, 3, 15)
        c4_range = np.linspace(-0.2, 0.2, 10)
        c8_range = np.linspace(-0.01, 0.01, 8)
        
        for c0v in c0_range:
            if time.time() > deadline:
                break
            for c4v in c4_range:
                if time.time() > deadline:
                    break
                for c8v in c8_range:
                    if time.time() > deadline:
                        break
                    val = objective(np.array([c0v, c4v, c8v]))
                    if val < best_val:
                        best_val = val
                        best_coeffs = np.array([c0v, c4v, c8v])
    
    # Refine around best with local optimization
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective, best_coeffs, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array([best_coeffs[0], best_coeffs[1], best_coeffs[2]])
# EVOLVE-BLOCK-END
