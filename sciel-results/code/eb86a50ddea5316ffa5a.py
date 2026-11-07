"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import sympy as sp

# Pre-compute the Hermite polynomials and their values at 0
# H_n(x) are physicists' Hermite polynomials
x = sp.Symbol('x')
H0 = sp.hermite(0, x)
H4 = sp.hermite(4, x)
H8 = sp.hermite(8, x)
H12 = sp.hermite(12, x)

# Values at 0
H0_0 = H0.subs(x, 0)
H4_0 = H4.subs(x, 0)
H8_0 = H8.subs(x, 0)
H12_0 = H12.subs(x, 0)

# Convert to lambdas for fast evaluation
# We need to evaluate the polynomial q(x) = g(x)/x^2
# g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0) / H12_0

# Let's get the coefficients of the polynomials for faster numerical evaluation
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 84x^10 + 2100x^8 - 25200x^6 + 138600x^4 - 332640x^2 + 10395

# Coefficients of H_n(x) = sum a_k x^{2k}
# H0: [1]
# H4: [3, -6, 1] (for x^0, x^2, x^4)
# H8: [105, -420, 210, -28, 1]
# H12: [10395, -332640, 138600, -25200, 2100, -84, 1]

c_H0 = np.array([1.0])
c_H4 = np.array([3.0, -6.0, 1.0])
c_H8 = np.array([105.0, -420.0, 210.0, -28.0, 1.0])
c_H12 = np.array([10395.0, -332640.0, 138600.0, -25200.0, 2100.0, -84.0, 1.0])

H0_0_val = 1.0
H4_0_val = 3.0
H8_0_val = 105.0
H12_0_val = 10395.0

def compute_value(c0, c1, c2):
    c3 = -(c0 * H0_0_val + c1 * H4_0_val + c2 * H8_0_val) / H12_0_val
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # Coefficients of g(x) in powers of x^2:
    # deg 0: c0*1 + c1*3 + c2*105 + c3*10395
    # deg 2: c1*(-6) + c2*(-420) + c3*(-332640)
    # deg 4: c1*1 + c2*210 + c3*138600
    # deg 6: c2*(-28) + c3*(-25200)
    # deg 8: c2*1 + c3*2100
    # deg 10: c3*(-84)
    # deg 12: c3*1
    
    # q(x) = g(x) / x^2
    # So q(x) is a polynomial in y = x^2 of degree 10.
    # q(y) = b10*y^10 + b9*y^9 + ... + b1
    # where b_k is the coefficient of x^{2k} in g(x).
    
    b12 = c3
    b10 = -84 * c3
    b8 = 2100 * c3
    b6 = -25200 * c3 - 28 * c2
    b4 = 138600 * c3 + 210 * c2 + c1
    b2 = -332640 * c3 - 420 * c2 - 6 * c1
    b0 = 10395 * c3 + 105 * c2 + 3 * c1 + c0
    
    # q(y) = b12*y^10 + b10*y^9 + b8*y^8 + b6*y^7 + b4*y^6 + b2*y^5 + b0*y^4 
    # Wait, g(x) has terms x^0, x^2, ..., x^12.
    # q(x) = g(x)/x^2 = c3*x^10 + c3*(-84)*x^8 + ... + c0*x^{-2}? No.
    # g(x) is even. g(0)=0 implies the x^0 term is 0.
    # So g(x) = a2*x^2 + a4*x^4 + ... + a12*x^12.
    # q(x) = a2 + a4*x^2 + ... + a12*x^10.
    # Let y = x^2. Q(y) = a2 + a4*y + ... + a12*y^10.
    # We need the largest real root r of q(x) where sign changes.
    # This corresponds to the largest positive real root R of Q(y) where sign changes.
    # r = sqrt(R).
    
    # Coefficients of Q(y):
    # Q(y) = b12*y^10 + b10*y^9 + b8*y^8 + b6*y^7 + b4*y^6 + b2*y^5 + b0*y^4
    # Wait, b0 is the x^0 coefficient of g(x), which is 0.
    # So the constant term of Q(y) is the x^2 coefficient of g(x), which is b2.
    # Let's re-index.
    # g(x) = g0 + g2*x^2 + g4*x^4 + g6*x^6 + g8*x^8 + g10*x^10 + g12*x^12
    # g0 = 0
    # q(x) = g2 + g4*x^2 + g6*x^4 + g8*x^6 + g10*x^8 + g12*x^10
    # Let y = x^2. Q(y) = g2 + g4*y + g6*y^2 + g8*y^3 + g10*y^4 + g12*y^5
    # We need the largest positive root R of Q(y) where sign changes.
    # Then r = sqrt(R).
    
    g12 = c3
    g10 = -84 * c3
    g8 = 2100 * c3
    g6 = -25200 * c3 - 28 * c2
    g4 = 138600 * c3 + 210 * c2 + c1
    g2 = -332640 * c3 - 420 * c2 - 6 * c1
    
    # Q(y) = g12*y^5 + g10*y^4 + g8*y^3 + g6*y^2 + g4*y + g2
    coeffs_Q = [g12, g10, g8, g6, g4, g2]
    
    # Find roots of Q(y)
    roots_y = np.roots(coeffs_Q)
    
    # Filter for real, positive roots
    real_roots = []
    for root in roots_y:
        if np.isreal(root):
            r_val = np.real(root)
            if r_val > 1e-9: # Must be positive for x to be real non-zero
                # Check sign change
                # Evaluate Q(y) at r_val - eps and r_val + eps
                eps = 1e-7
                y1 = r_val - eps
                y2 = r_val + eps
                q1 = np.polyval(coeffs_Q, y1)
                q2 = np.polyval(coeffs_Q, y2)
                if q1 * q2 < 0:
                    real_roots.append(r_val)
    
    if not real_roots:
        # If no sign-changing positive root, the behavior is undefined or infinite.
        # Return a large value to discourage this.
        return 1e10
        
    R_max = max(real_roots)
    r_x = np.sqrt(R_max)
    value = (r_x ** 2) / (2 * np.pi)
    return value

def construct():
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Calculate seed value
    seed_val = compute_value(seed[0], seed[1], seed[2])
    
    # Local search
    # We use Nelder-Mead to minimize the value
    # Bounds are not strictly necessary for Nelder-Mead, but we can use L-BFGS-B if we had bounds.
    # Let's try Nelder-Mead with a small initial step size.
    
    def objective(coeffs):
        return compute_value(coeffs[0], coeffs[1], coeffs[2])
        
    res = minimize(objective, seed, method='Nelder-Mead', options={'xatol': 1e-9, 'fatol': 1e-12, 'maxiter': 1000})
    
    if res.success and res.fun < seed_val:
        return res.x
    else:
        return seed
# EVOLVE-BLOCK-END
