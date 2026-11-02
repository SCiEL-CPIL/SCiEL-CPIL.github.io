"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_matrices():
    """Compute Gram matrices for L2 norms of g, xg, and g' (which relates to x*g_hat)."""
    # Hermite polynomials H_0, H_4, H_8, H_12
    # H_0 = 1
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H_12 = 4096x^12 - 73728x^10 + 516096x^8 - 1632960x^6 + 2402400x^4 - 1597440x^2 + 103950
    
    def H0(x): return np.ones_like(x)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    def H12(x): return 4096*x**12 - 73728*x**10 + 516096*x**8 - 1632960*x**6 + 2402400*x**4 - 1597440*x**2 + 103950
    
    # Gaussian weight
    def gauss(x): return np.exp(-x**2/2)
    
    # Use Gauss-Hermite quadrature for integration
    # np.polynomial.hermite.hermgauss gives nodes and weights for ∫ e^{-x^2} f(x) dx
    # But we have e^{-x^2/2}, so use a different approach
    # Let's use numerical integration with a fine grid
    x = np.linspace(-15, 15, 20000)
    dx = x[1] - x[0]
    w = gauss(x)
    
    H_funcs = [H0, H4, H8, H12]
    n = 4
    
    # Compute ||g||^2 = ∫ (Σ cᵢ Hᵢ e^{-x²/2})² dx = ∫ (Σ cᵢ Hᵢ)² e^{-x²} dx
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i,j] = np.sum(H_funcs[i](x) * H_funcs[j](x) * w**2) * dx
    
    # Compute ||xg||^2 = ∫ x² (Σ cᵢ Hᵢ e^{-x²/2})² dx = ∫ x² (Σ cᵢ Hᵢ)² e^{-x²} dx
    X = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            X[i,j] = np.sum(x**2 * H_funcs[i](x) * H_funcs[j](x) * w**2) * dx
    
    # Compute ||g'||^2 where g = (Σ cᵢ Hᵢ) e^{-x²/2}
    # g' = (Σ cᵢ Hᵢ') e^{-x²/2} + (Σ cᵢ Hᵢ)(-x) e^{-x²/2}
    # = [(Σ cᵢ Hᵢ') - x(Σ cᵢ Hᵢ)] e^{-x²/2}
    # ||g'||^2 = ∫ [(Σ cᵢ Hᵢ') - x(Σ cᵢ Hᵢ)]² e^{-x²} dx
    # = ∫ (Σ cᵢ Hᵢ')² e^{-x²} dx - 2∫ x(Σ cᵢ Hᵢ')(Σ cⱼ Hⱼ) e^{-x²} dx + ∫ x²(Σ cⱼ Hⱼ)² e^{-x²} dx
    
    # H_n' = 2n H_{n-1}
    def H0_prime(x): return np.zeros_like(x)
    def H4_prime(x): return 8 * H_funcs[0](x) * 4  # 4*H_3... wait
    # Actually H_n' = 2n H_{n-1}
    # H_0' = 0
    # H_4' = 8 H_3
    # H_8' = 16 H_7
    # H_12' = 24 H_11
    
    # Let me compute derivatives numerically instead for simplicity
    H_primes = []
    for i in range(n):
        H_primes.append(np.gradient(H_funcs[i](x), x))
    
    # ||g'||^2 = ∫ [(Σ cᵢ Hᵢ') - x(Σ cᵢ Hᵢ)]² e^{-x²} dx
    # Expand: let A = Σ cᵢ Hᵢ', B = x Σ cᵢ Hᵢ
    # = ∫ (A² - 2AB + B²) e^{-x²} dx
    # = c^T M c - 2 c^T N c + c^T X c
    # where M[i,j] = ∫ Hᵢ' Hⱼ' e^{-x²} dx
    #       N[i,j] = ∫ x Hᵢ' Hⱼ e^{-x²} dx
    #       X[i,j] = ∫ x² Hᵢ Hⱼ e^{-x²} dx (already computed)
    
    M = np.zeros((n, n))
    N = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i,j] = np.sum(H_primes[i] * H_primes[j] * w**2) * dx
            N[i,j] = np.sum(x * H_primes[i] * H_funcs[j](x) * w**2) * dx
    
    # D = M - 2N + X is the Gram matrix for ||g'||^2
    D = M - 2*N + X
    
    return G, X, D

def _compute_bound(coeffs):
    """Compute the uncertainty bound for given coefficients [a0, a4, a8]."""
    a0, a4, a8 = coeffs
    # H12 coefficient to make g(0) = 0
    # g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * e^0
    # H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 103950
    H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 1680.0, 103950.0
    a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    
    c = np.array([a0, a4, a8, a12])
    
    G, X, D = _compute_matrices()
    
    # ||g||^2 = c^T G c
    norm_g2 = c @ G @ c
    if norm_g2 <= 0:
        return 1e10
    
    # ||xg||^2 = c^T X c
    norm_xg2 = c @ X @ c
    if norm_xg2 < 0:
        norm_xg2 = 0
    
    # ||g'||^2 = c^T D c (this equals ||x*g_hat||^2 by Plancherel and F[x*f] = i*F[f]')
    norm_gprime2 = c @ D @ c
    if norm_gprime2 < 0:
        norm_gprime2 = 0
    
    # Uncertainty product bound: sqrt(norm_xg2/norm_g2) * sqrt(norm_gprime2/norm_g2)
    # = sqrt(norm_xg2 * norm_gprime2) / norm_g2
    bound = np.sqrt(max(norm_xg2, 0) * max(norm_gprime2, 0)) / norm_g2
    
    return bound

def _optimize_with_deadline(deadline):
    """Optimize the coefficients with a deadline guard."""
    start_time = time.time()
    
    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try multiple initial points
    initial_points = [
        x0,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.04, 0.0002]),
        np.array([-0.9, 0.06, 0.0004]),
        np.array([-1.5, 0.03, 0.0001]),
        np.array([-0.5, 0.08, 0.0005]),
    ]
    
    best_bound = _compute_bound(x0)
    best_coeffs = x0.copy()
    
    for x_init in initial_points:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _compute_bound,
                x_init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Try BFGS with gradient approximation if time permits
    if time.time() < deadline - 60:
        try:
            result = minimize(
                _compute_bound,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill
    
    try:
        coeffs = _optimize_with_deadline(deadline)
        return np.array(coeffs)
    except Exception:
        # Fallback to seed values
        return np.array([-113/100, 1/25, 1/3240])
# EVOLVE-BLOCK-END
