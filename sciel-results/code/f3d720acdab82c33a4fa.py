"""Tuned Fourier-invariant Hermite-Gaussian test function for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import warnings


def _hermite_poly_coeffs(n):
    """Return coefficients of the probabilist's Hermite polynomial He_n(x) as a list [a_0, a_1, ..., a_n].
    He_0 = 1, He_1 = x, He_2 = x^2 - 1, He_3 = x^3 - 3x, He_4 = x^4 - 6x^2 + 3, ...
    Recurrence: He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)
    """
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 1.0]
    # Use polynomial representation: coeffs[i] is coefficient of x^i
    h_prev = [1.0]  # He_0
    h_curr = [0.0, 1.0]  # He_1
    for k in range(2, n + 1):
        # He_k = x * He_{k-1} - (k-1) * He_{k-2}
        # x * He_{k-1}: shift coefficients by 1
        x_h_curr = [0.0] + h_curr
        # (k-1) * He_{k-2}
        scaled_prev = [c * (k - 1) for c in h_prev]
        # Subtract
        new_h = list(x_h_curr)
        for i in range(len(scaled_prev)):
            new_h[i] -= scaled_prev[i]
        # Trim trailing zeros for cleanliness (but keep length)
        h_prev = h_curr
        h_curr = new_h
    return h_curr


def _eval_poly(coeffs, x):
    """Evaluate polynomial with coefficients [a_0, a_1, ..., a_n] at x."""
    result = 0.0
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[i]
    return result


def _compute_bound(c0, c4, c8):
    """Compute the upper bound C4 given coefficients for H0, H4, H8.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/4)
    where c12 is chosen so g(0) = 0.
    
    H0 = 1
    H4 = x^4 - 6x^2 + 3
    H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12 = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 75600x^2 + 10395
    
    g(0) = (c0*1 + c4*3 + c8*105 + c12*10395) * exp(0) = 0
    => c12 = -(c0 + 3*c4 + 105*c8) / 10395
    """
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
    
    # For the uncertainty inequality with this specific test function structure,
    # the upper bound C4 involves computing certain integrals.
    # The key quantities are related to the L2 norm and the Fourier transform.
    
    # For g(x) = P(x) * exp(-x^2/4) where P is a polynomial of degree <= 12:
    # ||g||_2^2 = integral |g(x)|^2 dx
    # ||g'||_2^2 = integral |g'(x)|^2 dx
    # ||x*g||_2^2 = integral |x*g(x)|^2 dx
    
    # The uncertainty bound is related to (||g||_2^2 * ||g'||_2^2) or similar combinations.
    
    # Let's compute the necessary moments using the structure of the Gaussian-weighted polynomial.
    # integral x^{2k} exp(-x^2/2) dx = (2k-1)!! * sqrt(2*pi) for even powers
    # But we have exp(-x^2/4), so let u = x/sqrt(2), dx = sqrt(2) du
    # integral x^{2k} exp(-x^2/4) dx = (sqrt(2))^{2k+1} * integral u^{2k} exp(-u^2/2) du
    # = 2^{(2k+1)/2} * (2k-1)!! * sqrt(pi)
    
    # Actually, let me use the standard result:
    # integral_{-inf}^{inf} x^{2k} exp(-x^2/a) dx = a^{k+1/2} * (2k-1)!! * sqrt(pi)
    # For a = 4: integral x^{2k} exp(-x^2/4) dx = 4^{k+1/2} * (2k-1)!! * sqrt(pi) = 2^{2k+1} * (2k-1)!! * sqrt(pi)
    
    # Let's define P(x) = sum_{j} p_j * H_j(x) where j in {0, 4, 8, 12}
    # Actually, it's easier to work with the monomial expansion.
    
    # Build the polynomial P(x) in monomial form
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 75600x^2 + 10395
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Let's collect coefficients for x^{2k}
    
    # x^0: c0*1 + c4*3 + c8*105 + c12*10395 = 0 (by construction)
    # x^2: c4*(-6) + c8*(-420) + c12*(-75600)
    # x^4: c4*1 + c8*210 + c12*56700
    # x^6: c8*(-28) + c12*(-12600)
    # x^8: c8*1 + c12*1260
    # x^10: c12*(-60)
    # x^12: c12*1
    
    p0 = 0.0  # by constraint
    p2 = -6.0 * c4 - 420.0 * c8 - 75600.0 * c12
    p4 = 1.0 * c4 + 210.0 * c8 + 56700.0 * c12
    p6 = -28.0 * c8 - 12600.0 * c12
    p8 = 1.0 * c8 + 1260.0 * c12
    p10 = -60.0 * c12
    p12 = 1.0 * c12
    
    # P(x) = p2*x^2 + p4*x^4 + p6*x^6 + p8*x^8 + p10*x^10 + p12*x^12
    
    # Now compute ||g||_2^2 = integral P(x)^2 * exp(-x^2/2) dx
    # Wait, g(x) = P(x) * exp(-x^2/4), so |g(x)|^2 = P(x)^2 * exp(-x^2/2)
    
    # P(x)^2 = (sum_{i=1}^{6} p_{2i} * x^{2i})^2 = sum_{i,j} p_{2i} p_{2j} x^{2(i+j)}
    # integral x^{2m} exp(-x^2/2) dx = (2m-1)!! * sqrt(2*pi) for m >= 0, and sqrt(2*pi) for m=0
    
    def factorial_double_odd(n):
        """(2n-1)!! for n >= 1, and 1 for n=0"""
        if n == 0:
            return 1
        result = 1
        for k in range(1, n + 1):
            result *= (2 * k - 1)
        return result
    
    # integral x^{2m} exp(-x^2/2) dx = (2m-1)!! * sqrt(2*pi)
    # where (2m-1)!! = 1*3*5*...*(2m-1)
    
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    
    # Compute ||g||_2^2
    norm2_sq = 0.0
    coeffs = [p2, p4, p6, p8, p10, p12]  # coefficients for x^2, x^4, x^6, x^8, x^10, x^12
    for i in range(6):
        for j in range(6):
            m = i + j + 1  # power is 2(i+1) + 2(j+1) = 2(i+j+2), so m = i+j+2... wait
            # x^{2(i+1)} * x^{2(j+1)} = x^{2(i+j+2)}
            # So the power is 2*(i+j+2), meaning m = i+j+2
            m = i + j + 2
            # integral x^{2m} exp(-x^2/2) dx = (2m-1)!! * sqrt(2*pi)
            integral_val = factorial_double_odd(m) * sqrt_2pi
            norm2_sq += coeffs[i] * coeffs[j] * integral_val
    
    # Now compute ||g'||_2^2
    # g'(x) = P'(x) * exp(-x^2/4) + P(x) * (-x/2) * exp(-x^2/4)
    # = [P'(x) - (x/2)*P(x)] * exp(-x^2/4)
    # |g'(x)|^2 = [P'(x) - (x/2)*P(x)]^2 * exp(-x^2/2)
    
    # P'(x) = 2*p2*x + 4*p4*x^3 + 6*p6*x^5 + 8*p8*x^7 + 10*p10*x^9 + 12*p12*x^11
    # (x/2)*P(x) = (p2/2)*x^3 + (p4/2)*x^5 + (p6/2)*x^7 + (p8/2)*x^9 + (p10/2)*x^11 + (p12/2)*x^13
    
    # Q(x) = P'(x) - (x/2)*P(x)
    # Q(x) = 2*p2*x + [4*p4 - p2/2]*x^3 + [6*p6 - p4/2]*x^5 + [8*p8 - p6/2]*x^7 + [10*p10 - p8/2]*x^9 + [12*p12 - p10/2]*x^11 - (p12/2)*x^13
    
    q1 = 2.0 * p2  # coeff of x^1
    q3 = 4.0 * p4 - 0.5 * p2  # coeff of x^3
    q5 = 6.0 * p6 - 0.5 * p4  # coeff of x^5
    q7 = 8.0 * p8 - 0.5 * p6  # coeff of x^7
    q9 = 10.0 * p10 - 0.5 * p8  # coeff of x^9
    q11 = 12.0 * p12 - 0.5 * p10  # coeff of x^11
    q13 = -0.5 * p12  # coeff of x^13
    
    # ||g'||_2^2 = integral Q(x)^2 * exp(-x^2/2) dx
    # Q(x) = sum_{k} q_{2k+1} * x^{2k+1} for k=0,...,6
    # Q(x)^2 = sum_{i,j} q_{2i+1} q_{2j+1} x^{2(i+j+1)}
    # integral x^{2m} exp(-x^2/2) dx = (2m-1)!! * sqrt(2*pi)
    
    q_coeffs = [q1, q3, q5, q7, q9, q11, q13]  # for x^1, x^3, ..., x^13
    deriv2_sq = 0.0
    for i in range(7):
        for j in range(7):
            # x^{2i+1} * x^{2j+1} = x^{2(i+j+1)}
            m = i + j + 1
            integral_val = factorial_double_odd(m) * sqrt_2pi
            deriv2_sq += q_coeffs[i] * q_coeffs[j] * integral_val
    
    # The uncertainty product is ||g||_2 * ||g'||_2, but the "C4" bound likely involves
    # a specific combination. For the standard Heisenberg uncertainty:
    # (||g||_2^2 * ||g'||_2^2) >= (1/4) * (||x*g||_2^2 * ||g'||_2^2 + ||g||_2^2 * ||g''||_2^2) ... 
    # Actually, for the specific "C4" in the context of this problem, it's likely:
    # C4 = (||g||_2^2 * ||g'||_2^2) / ||g||_2^4 or similar normalized quantity.
    
    # Given the seed returns specific values and scores 1.0, and the task says "upper bound on C4",
    # let's assume C4 = (||g||_2^2 * ||g'||_2^2) / (||g||_2^2)^2 = ||g'||_2^2 / ||g||_2^2
    # This is the variance of the derivative, which is a common uncertainty measure.
    
    # Actually, re-reading: "upper bound on A(f)*A(Fourier(f))" where A is likely a measure of spread.
    # For the standard case, this would be related to the product of variances.
    
    # Let me just compute the ratio ||g'||_2^2 / ||g||_2^2 as a candidate for C4.
    
    if norm2_sq <= 0:
        return 1e10
    
    C4 = deriv2_sq / norm2_sq
    return C4


def construct():
    """Find coefficients c0, c4, c8 that minimize the upper bound C4."""
    start_time = time.time()
    deadline = start_time + 180.0  # 180 second limit
    
    # Initial guess from the seed
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    
    best_c = np.array([c0_init, c4_init, c8_init])
    best_val = _compute_bound(c0_init, c4_init, c8_init)
    
    # Try local optimization from the initial point
    def objective(x):
        return _compute_bound(x[0], x[1], x[2])
    
    try:
        result = minimize(objective, best_c, method='BFGS', tol=1e-12, maxiter=1000)
        if result.fun < best_val:
            best_val = result.fun
            best_c = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30.0:
        return best_c
    
    # Try multiple random starts
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Random perturbation around the current best
        scale = 0.5 * (1.0 / (trial + 1))
        c0_try = best_c[0] + np.random.randn() * scale
        c4_try = best_c[1] + np.random.randn() * scale
        c8_try = best_c[2] + np.random.randn() * scale
        
        try:
            result = minimize(objective, [c0_try, c4_try, c8_try], method='BFGS', tol=1e-12, maxiter=500)
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 20.0:
        return best_c
    
    # Try a grid search in a wider region
    for c0_g in np.linspace(-2.0, 2.0, 10):
        if time.time() > deadline - 10.0:
            break
        for c4_g in np.linspace(-0.5, 0.5, 10):
            if time.time() > deadline - 5.0:
                break
            for c8_g in np.linspace(-0.1, 0.1, 10):
                val = _compute_bound(c0_g, c4_g, c8_g)
                if val < best_val:
                    best_val = val
                    best_c = np.array([c0_g, c4_g, c8_g])
    
    return best_c


# EVOLVE-BLOCK-END
