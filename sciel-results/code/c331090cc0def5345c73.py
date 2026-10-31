"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial

# Deadline guard
_DEADLINE = 210.0  # seconds
_START_TIME = time.time()

def _check_deadline():
    return (time.time() - _START_TIME) < (_DEADLINE - 10.0)  # 10s safety margin

def _hermite(n, x):
    """Probabilists' Hermite polynomial H_n(x) (physicists' convention scaled)."""
    # We use physicists' Hermite polynomials: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
    # But for the Fourier-invariant setup, we typically work with the standard Hermite functions.
    # The task mentions H0, H4, H8, H12. Let's use the standard physicists' Hermite polynomials.
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 4x^2 - 2
    # H_3 = 8x^3 - 12x
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H_12 = 4096x^12 - 98304x^10 + 943712x^8 - 4718592x^6 + 10321920x^4 - 10813440x^2 + 443520
    
    # Let's compute them using scipy or manually.
    # Actually, let's use the recurrence relation or just hardcode the needed ones.
    # For efficiency, I'll hardcode the polynomial coefficients for H_0, H_4, H_8, H_12.
    
    if n == 0:
        return np.ones_like(x)
    elif n == 4:
        return 16*x**4 - 48*x**2 + 12
    elif n == 8:
        return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    elif n == 12:
        return 4096*x**12 - 98304*x**10 + 943712*x**8 - 4718592*x**6 + 10321920*x**4 - 10813440*x**2 + 443520
    else:
        raise ValueError(f"Unsupported Hermite degree: {n}")

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the test function
    g(x) = (c0*H_0 + c4*H_4 + c8*H_8 + c12*H_12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    The uncertainty bound for a function g is related to the variance of |g|^2
    and the variance of |F(g)|^2 (in the Fourier domain).
    
    For a Gaussian-modulated Hermite polynomial, the Fourier transform is also
    a Gaussian-modulated Hermite polynomial (possibly with a phase factor).
    
    Specifically, if g(x) = P(x) * exp(-x^2/2), then 
    F(g)(xi) = (i)^k * P(xi) * exp(-xi^2/2) for appropriate k, but actually
    the Fourier transform of H_n(x) * exp(-x^2/2) is (i)^n * H_n(xi) * exp(-xi^2/2).
    
    So |F(g)|^2 = |P(xi)|^2 * exp(-xi^2), and |g|^2 = |P(x)|^2 * exp(-x^2).
    
    The "A" quantity is typically the L^1 norm or the area under |g|^2.
    A(g) = integral |g(x)|^2 dx = integral |P(x)|^2 exp(-x^2) dx.
    A(F(g)) = integral |F(g)(xi)|^2 dxi = integral |P(xi)|^2 exp(-xi^2) dxi = A(g).
    
    So A(g) = A(F(g)), and the product is A(g)^2.
    
    Wait, the task says "upper bound on A(f)*A(Fourier(f))". This is likely the
    Heisenberg uncertainty principle bound, which involves variances.
    
    Let me reconsider. The standard uncertainty principle states:
    sigma_x^2 * sigma_xi^2 >= 1/4
    where sigma_x^2 = integral (x - mu_x)^2 |g(x)|^2 dx / integral |g(x)|^2 dx
    and sigma_xi^2 = integral (xi - mu_xi)^2 |F(g)(xi)|^2 dxi / integral |F(g)(xi)|^2 dxi
    
    But the task mentions A(f)*A(Fourier(f)), which might be different.
    
    Actually, looking at the context of "Fourier-invariant Hermite-polynomial times Gaussian",
    this is likely referring to the uncertainty bound in the context of the 
    Beals-Hejazi or similar inequality, where A(f) might be the L^1 norm or 
    the area of the support, or it could be the product of the variances.
    
    Given the mention of "upper bound on C4", this might be a specific constant.
    
    Let me assume that the quantity to minimize is the product of the variances:
    sigma_x^2 * sigma_xi^2
    
    For g(x) = P(x) * exp(-x^2/2), with P(x) = c0*H_0 + c4*H_4 + c8*H_8 + c12*H_12
    
    First, compute c12 such that g(0) = 0:
    P(0) = c0*H_0(0) + c4*H_4(0) + c8*H_8(0) + c12*H_12(0) = 0
    
    H_0(0) = 1
    H_4(0) = 12
    H_8(0) = 1680
    H_12(0) = 443520
    
    So: c0 + 12*c4 + 1680*c8 + 443520*c12 = 0
    => c12 = -(c0 + 12*c4 + 1680*c8) / 443520
    
    Now, to compute the variances, we need:
    integral |P(x)|^2 exp(-x^2) dx
    integral x^2 |P(x)|^2 exp(-x^2) dx
    
    Since P(x) is a real polynomial, |P(x)|^2 = P(x)^2.
    
    P(x)^2 = (c0*H_0 + c4*H_4 + c8*H_8 + c12*H_12)^2
    
    This is a polynomial of degree 24. We can expand it and integrate term by term
    against exp(-x^2).
    
    The integral of x^(2k) * exp(-x^2) dx from -inf to inf is sqrt(pi) * (2k-1)!! / 2^k
    for even powers, and 0 for odd powers.
    
    (2k-1)!! = (2k)! / (2^k * k!)
    
    So integral x^(2k) exp(-x^2) dx = sqrt(pi) * (2k)! / (2^(2k) * k!)
    
    Let me compute this numerically for efficiency.
    
    Actually, let me just compute the integrals numerically using Gauss-Hermite quadrature
    or direct numerical integration.
    """
    # Compute c12
    c12 = -(c0 + 12*c4 + 1680*c8) / 443520.0
    
    # Define P(x)
    def P(x):
        return c0*_hermite(0, x) + c4*_hermite(4, x) + c8*_hermite(8, x) + c12*_hermite(12, x)
    
    # Compute integrals using Gauss-Hermite quadrature
    # For integral f(x) * exp(-x^2) dx, use Gauss-Hermite with weight exp(-x^2)
    # scipy.integrate.gauss_hermite gives nodes and weights for weight exp(-x^2)
    
    # Use a high-order Gauss-Hermite quadrature
    n_quad = 100
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    
    # P(x) at the nodes
    P_vals = P(x_nodes)
    P_sq = P_vals**2
    
    # integral |P|^2 exp(-x^2) dx
    I0 = np.sum(w_nodes * P_sq)
    
    # integral x^2 |P|^2 exp(-x^2) dx
    I2 = np.sum(w_nodes * x_nodes**2 * P_sq)
    
    # Variance in x-space: sigma_x^2 = (I2 - mu_x^2) / I0
    # Since P is a combination of even Hermite polynomials (H_0, H_4, H_8, H_12 are all even),
    # P(x) is an even function, so mu_x = 0.
    # Thus sigma_x^2 = I2 / I0
    
    # For the Fourier transform, F(g)(xi) = (i)^n * H_n(xi) * exp(-xi^2/2) for each term,
    # but since we have a combination, and the Fourier transform is linear:
    # F(g)(xi) = [c0*H_0(xi) + c4*(i)^4*H_4(xi) + c8*(i)^8*H_8(xi) + c12*(i)^12*H_12(xi)] * exp(-xi^2/2)
    # (i)^4 = 1, (i)^8 = 1, (i)^12 = (i)^4*(i)^8 = 1
    # So F(g)(xi) = P(xi) * exp(-xi^2/2)
    # Thus |F(g)|^2 = P(xi)^2 * exp(-xi^2)
    # And the variance in the Fourier domain is the same as in the x-domain!
    # sigma_xi^2 = sigma_x^2
    
    # So the product is sigma_x^2 * sigma_xi^2 = (I2/I0)^2
    
    if I0 <= 0:
        return np.inf
    
    sigma_x_sq = I2 / I0
    return sigma_x_sq**2

def construct():
    """
    Find the coefficients c0, c4, c8 that minimize the uncertainty bound.
    """
    # Initial guess from the current program
    c0_init, c4_init, c8_init = -113/100, 1/25, 1/3240
    
    # Define the objective function
    def objective(params):
        return _compute_bound(params[0], params[1], params[2])
    
    # Multi-start local optimization
    best_val = np.inf
    best_params = np.array([c0_init, c4_init, c8_init])
    
    # First, try the initial guess
    val0 = objective(best_params)
    if val0 < best_val:
        best_val = val0
    
    # Generate multiple random starting points
    np.random.seed(42)
    n_starts = 50
    
    for i in range(n_starts):
        if not _check_deadline():
            break
        
        # Random start in a reasonable range
        c0_r = np.random.uniform(-5, 5)
        c4_r = np.random.uniform(-1, 1)
        c8_r = np.random.uniform(-0.1, 0.1)
        
        x0 = np.array([c0_r, c4_r, c8_r])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Also try differential evolution for global search if time permits
    if _check_deadline():
        try:
            bounds = [(-10, 10), (-2, 2), (-0.5, 0.5)]
            de_result = differential_evolution(objective, bounds, maxiter=100, 
                                               popsize=15, seed=42, tol=1e-10,
                                               polish=True)
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    # Final polish with tight tolerance
    if _check_deadline():
        try:
            result = minimize(objective, best_params, method='L-BFGS-B',
                             options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
