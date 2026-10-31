"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial
from math import sqrt, pi


def _hermite_values_at_zero(max_n=24):
    """Compute H_n(0) for even n up to max_n."""
    h0 = np.zeros(max_n + 1)
    for n in range(0, max_n + 1, 2):
        # H_n(0) for even n: (-1)^(n/2) * (n-1)!!
        # (n-1)!! = (n-1)(n-3)...1 = n! / (2^(n/2) * (n/2)!)
        # So H_n(0) = (-1)^(n/2) * n! / (2^(n/2) * (n/2)!)
        h0[n] = (-1) ** (n // 2) * factorial(n) / (2 ** (n // 2) * factorial(n // 2))
    return h0


def construct():
    t0 = time.time()
    deadline = t0 + 170  # Safety margin: 170s out of 180s
    
    # H_n(0) values for even n
    h0 = _hermite_values_at_zero(12)
    h00, h04, h08, h012 = h0[0], h0[4], h0[8], h0[12]
    
    # Constraint: g(0) = 0
    # g(0) = e^0 * (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) = 0
    # So: a12 = -(a0*h00 + a4*h04 + a8*h08) / h012
    
    # The test function is g(x) = e^{-x^2/2} * P(x) where P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # We want to minimize the upper bound on C4 = A(g) * A(F(g))
    # where A(f) = ||f||_1 / ||f||_2
    
    # For Hermite functions, we can compute the L1 and L2 norms analytically/numerically
    # in the Hermite basis.
    
    # The Hermite functions psi_n(x) = (2^n n! sqrt(pi))^{-1/2} e^{-x^2/2} H_n(x) form an orthonormal basis.
    # So g(x) = sum c_k psi_{2k}(x) where c_k = a_k * (2^{2k} (2k)! sqrt(pi))^{1/2}
    
    # L2 norm: ||g||_2^2 = sum c_k^2 (by orthonormality)
    # L1 norm: ||g||_1 = integral |g(x)| dx - this needs numerical computation
    
    # For the Fourier transform, F(psi_n)(xi) = i^n psi_n(xi) (up to a constant)
    # So F(g)(xi) = sum c_k i^{2k} psi_{2k}(xi) = sum c_k (-1)^k psi_{2k}(xi)
    # (since i^{2k} = (-1)^k for integer k)
    
    # The key insight: we can express the objective as a function of the coefficients
    # and use numerical optimization with a well-conditioned parameterization.
    
    # Let's use a grid to compute the L1 norms accurately
    x = np.linspace(-20, 20, 20000)
    dx = x[1] - x[0]
    
    from scipy.special import eval_hermite
    
    # Precompute basis functions
    def psi_n(x, n):
        norm = sqrt(2 ** n * factorial(n) * sqrt(pi))
        H_n = eval_hermite(n, x)
        return (1.0 / norm) * np.exp(-x ** 2 / 2) * H_n
    
    psi_0 = psi_n(x, 0)
    psi_4 = psi_n(x, 4)
    psi_8 = psi_n(x, 8)
    psi_12 = psi_n(x, 12)
    
    # Values at x=0 for the constraint
    idx_0 = np.argmin(np.abs(x))
    p0_0 = psi_0[idx_0]
    p0_4 = psi_4[idx_0]
    p0_8 = psi_8[idx_0]
    p0_12 = psi_12[idx_0]
    
    def compute_objective(a0, a4, a8):
        """Compute the uncertainty product for given coefficients."""
        # Compute a12 from constraint g(0) = 0
        a12 = -(a0 * p0_0 + a4 * p0_4 + a8 * p0_8) / p0_12
        
        # Compute g(x) and F(g)(xi)
        g = a0 * psi_0 + a4 * psi_4 + a8 * psi_8 + a12 * psi_12
        
        # For the Fourier transform with convention F(f)(xi) = int f(x) e^{-ixxi} dx:
        # F(e^{-x^2/2} H_n(x)) = i^n e^{-xi^2/2} H_n(xi)
        # So F(g)(xi) = a0 * i^0 * psi_0(xi) + a4 * i^4 * psi_4(xi) + a8 * i^8 * psi_8(xi) + a12 * i^12 * psi_12(xi)
        # = a0 * psi_0(xi) + a4 * psi_4(xi) + a8 * psi_8(xi) + a12 * psi_12(xi)
        # (since i^0 = i^4 = i^8 = i^12 = 1)
        
        # Wait, that's not right. Let me reconsider the Fourier transform convention.
        # Actually, for the standard Hermite function Fourier transform:
        # F(psi_n)(xi) = (-i)^n psi_n(xi) for the convention F(f)(xi) = int f(x) e^{-2*pi*i*x*xi} dx
        # Or F(psi_n)(xi) = i^n psi_n(xi) for the convention F(f)(xi) = int f(x) e^{-i*x*xi} dx
        
        # Let's use the convention where F(psi_n) = i^n psi_n for even n
        # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = 1
        # So F(g)(xi) = a0 * psi_0(xi) + a4 * psi_4(xi) + a8 * psi_8(xi) + a12 * psi_12(xi)
        # = g(xi) (same function!)
        
        # This means ||F(g)||_1 = ||g||_1 and ||F(g)||_2 = ||g||_2
        # So A(F(g)) = A(g)
        
        # Therefore A(g) * A(F(g)) = A(g)^2 = (||g||_1 / ||g||_2)^2
        
        l2_sq = np.sum(g ** 2) * dx
        if l2_sq < 1e-12:
            return 1e10
        
        l1 = np.sum(np.abs(g)) * dx
        
        return (l1 / sqrt(l2_sq)) ** 2
    
    # Use the seed values as initial guess
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Try to optimize using scipy with multiple methods
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        a0, a4, a8 = params
        return compute_objective(a0, a4, a8)
    
    best_params = x0.copy()
    best_score = compute_objective(*x0)
    
    # Try multiple optimization methods
    methods = [
        ('Nelder-Mead', {'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}),
        ('L-BFGS-B', {'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}),
    ]
    
    for method, options in methods:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, best_params, method=method, options=options)
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    return np.array(best_params)

# EVOLVE-BLOCK-END
