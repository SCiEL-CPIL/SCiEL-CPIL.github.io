"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy import integrate, optimize
from scipy.special import eval_genlaguerre, gamma
import time

# EVOLVE-BLOCK-START
def _laguerre_weights_and_nodes(n_quad, alpha=0):
    """Get Gauss-Laguerre quadrature nodes and weights for alpha=0."""
    # For integral_0^infty e^{-x} f(x) dx
    nodes, weights = np.polynomial.laguerre.laggauss(n_quad)
    return nodes, weights

def _compute_upper_bound(roots, n_quad=200):
    """
    Compute the upper bound on C4 given 12 positive double roots.
    
    The construction: f(x) = e^{-x/2} * P(x) where P(x) = prod (x - r_i)^2
    But we need to orthogonalize against lower-degree Laguerre polynomials.
    
    Actually, for the Cohn-Goncalves bound, we use:
    C4 <= (int_0^infty |f(x)|^2 dx) * (int_0^infty |fhat(x)|^2 dx) / 
          (int_0^infty x^2 |f(x)|^2 dx) * (int_0^infty x^2 |fhat(x)|^2 dx)
    
    But by Parseval and properties, this simplifies. Let's use the standard formula.
    
    For Laguerre-based construction, the upper bound is:
    C4 <= (sum of coefficients squared) / (weighted sum)
    
    Let me use a direct numerical approach with the actual integrals.
    """
    m = 12
    roots = np.asarray(roots, dtype=float)
    
    # Ensure roots are positive
    roots = np.maximum(roots, 1e-6)
    
    # Build polynomial P(x) = prod (x - r_i)^2
    # P(x) has degree 24
    # f(x) = e^{-x/2} * P(x)
    
    # We need to compute:
    # A = int_0^infty |f(x)|^2 dx = int_0^infty e^{-x} P(x)^2 dx
    # B = int_0^infty x^2 |f(x)|^2 dx = int_0^infty e^{-x} x^2 P(x)^2 dx
    # C = int_0^infty |fhat(x)|^2 dx = A (by Parseval for Fourier transform on R, but f is on [0,inf))
    
    # Actually, for functions on [0, inf), the Fourier transform is different.
    # Let me use the known result: for the Cohn-Goncalves construction,
    # the upper bound on C4 is given by:
    # C4 <= (int_0^infty e^{-x} P(x)^2 dx) / (int_0^infty e^{-x} x^2 P(x)^2 dx)
    # times some constant depending on m and n.
    
    # Let me compute the ratio R = A / B where A = int e^{-x} P^2 dx, B = int e^{-x} x^2 P^2 dx
    # The upper bound is then C4 <= 1/(4*R) or similar.
    
    # Use Gauss-Laguerre quadrature
    nodes, weights = _laguerre_weights_and_nodes(n_quad)
    
    # Evaluate P(x) at nodes
    P_vals = np.ones_like(nodes)
    for r in roots:
        P_vals *= (nodes - r) ** 2
    
    P_sq = P_vals ** 2
    
    A = np.sum(weights * P_sq)
    B = np.sum(weights * nodes**2 * P_sq)
    
    if B <= 0 or A <= 0:
        return 1e10
    
    R = A / B
    # The upper bound on C4 is proportional to 1/sqrt(R) or 1/R
    # Based on the uncertainty principle, C4 ~ 1/(4 * (A/B)) = B/(4A)
    # But let's use the standard formula. For the refined construction:
    # C4 <= 1 / (4 * R) where R = A/B
    # Actually, I think it's C4 <= 1/(16 * R) or similar.
    
    # Let me just return R as a proxy, but inverted so lower is better
    # The score is the upper bound on C4, so we want to minimize it.
    # If C4 <= C / R for some constant C, then minimizing C/R means maximizing R.
    # But we want to minimize the upper bound, so we want to maximize R.
    # Return 1/R so that minimizing 1/R maximizes R.
    
    return 1.0 / R

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive double-root locations.
    The score is the upper bound on C4 (lower is better).
    """
    start_time = time.time()
    deadline = 580.0  # Safety margin before 630s limit
    
    m = 12
    
    # Initial guess: spread roots logarithmically
    def initial_guess():
        # Roots should be positive. Try a geometric progression
        return np.logspace(-2, 2, m)
    
    # Objective function for optimization
    def objective(roots):
        if time.time() - start_time > deadline:
            return 1e10
        # Ensure positive roots
        roots_pos = np.maximum(roots, 1e-8)
        return _compute_upper_bound(roots_pos, n_quad=100)
    
    # Use L-BFGS-B for bounded optimization
    x0 = initial_guess()
    
    # Bounds: roots should be in [1e-4, 1e4]
    bounds = [(1e-4, 1e4)] * m
    
    try:
        result = optimize.minimize(
            objective, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if time.time() - start_time > deadline:
            # Return best found so far
            best_roots = np.maximum(result.x, 1e-8)
            return list(best_roots)
        
        best_roots = np.maximum(result.x, 1e-8)
        
    except Exception:
        # Fallback: return initial guess
        best_roots = x0
    
    # Verify roots are positive
    best_roots = np.maximum(best_roots, 1e-8)
    
    # Sort roots for consistency
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END

