# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import warnings

# Suppress warnings from optimization or numerical issues
warnings.filterwarnings("ignore")

# Pre-computed coefficients for Hermite polynomials H_0, H_4, H_8, H_12
# H_n(x) are physicists' Hermite polynomials.
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 84x^10 + 2100x^8 - 25200x^6 + 138600x^4 - 332640x^2 + 10395

# Coefficients for powers of x: [c_0, c_2, c_4, c_6, c_8, c_10, c_12]
# Note: H_n(x) only has even powers.
# We represent polynomials as lists of coefficients for x^0, x^2, x^4, ...
# This allows us to easily form Q(y) where y=x^2.

# H_0: [1]
# H_4: [3, -6, 1] -> 3 - 6y + y^2
# H_8: [105, -420, 210, -28, 1] -> 105 - 420y + 210y^2 - 28y^3 + y^4
# H_12: [10395, -332640, 138600, -25200, 2100, -84, 1] -> 10395 - 332640y + ... + y^6

H0_coeffs = np.array([1.0])
H4_coeffs = np.array([3.0, -6.0, 1.0])
H8_coeffs = np.array([105.0, -420.0, 210.0, -28.0, 1.0])
H12_coeffs = np.array([10395.0, -332640.0, 138600.0, -25200.0, 2100.0, -84.0, 1.0])

# Values at 0 (constant term)
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

def compute_value(c0, c1, c2):
    """
    Computes the value r^2 / (2*pi) for the given coefficients.
    """
    # Calculate c3 such that g(0) = 0
    # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Construct g(x) as a polynomial in y = x^2.
    # g(x) = c0*H0(y) + c1*H4(y) + c2*H8(y) + c3*H12(y)
    # Since H12 has the highest degree (6 in y), we pad others to length 7.
    
    # Length of H12_coeffs is 7 (y^0 to y^6)
    # We create a zero array of length 7 and add scaled polynomials.
    g_coeffs = np.zeros(7)
    
    # Add c0 * H0
    g_coeffs[0:1] += c0 * H0_coeffs
    # Add c1 * H4
    g_coeffs[0:3] += c1 * H4_coeffs
    # Add c2 * H8
    g_coeffs[0:5] += c2 * H8_coeffs
    # Add c3 * H12
    g_coeffs[0:7] += c3 * H12_coeffs
    
    # g(0) should be 0. Due to floating point, it might not be exactly 0.
    # The constant term g_coeffs[0] corresponds to the x^0 term.
    # q(x) = g(x) / x^2.
    # If g(x) = a0 + a2*x^2 + a4*x^4 + ..., then q(x) = a2 + a4*x^2 + ...
    # In terms of y=x^2, Q(y) = a2 + a4*y + a6*y^2 + ...
    # So Q(y) coefficients are g_coeffs[1], g_coeffs[2], ..., g_coeffs[6]
    
    # Check if g(0) is close to 0. If not, the division is invalid or c3 calculation was wrong.
    # But c3 is calculated exactly to make it 0 in exact arithmetic.
    # In floating point, g_coeffs[0] might be ~1e-16. We ignore it.
    
    # Q(y) is a polynomial of degree 5 (since g is degree 12 in x, degree 6 in y, but constant term is 0, so degree 5 in y for Q).
    # Q(y) = g_coeffs[6]*y^5 + g_coeffs[5]*y^4 + g_coeffs[4]*y^3 + g_coeffs[3]*y^2 + g_coeffs[2]*y + g_coeffs[1]
    
    # np.roots expects coefficients in descending order of power.
    Q_coeffs = g_coeffs[6:0:-1]
    
    # Find roots of Q(y)
    try:
        roots = np.roots(Q_coeffs)
    except Exception:
        return 1e10
        
    # Filter for real, positive roots
    # We need the largest real root R where Q(y) changes sign.
    # A simple real root (multiplicity 1) always changes sign.
    # A double root (multiplicity 2) does not.
    
    real_pos_roots = []
    for r in roots:
        if abs(r.imag) < 1e-8: # Real root
            r_val = r.real
            if r_val > 1e-8: # Positive root
                real_pos_roots.append(r_val)
                
    if not real_pos_roots:
        return 1e10
        
    # Sort roots in descending order to find the largest
    real_pos_roots.sort(reverse=True)
    
    # We need to check for sign change.
    # For a polynomial, if the multiplicity of the root is odd, the sign changes.
    # If even, it doesn't.
    # We can check the derivative Q'(y) at the root.
    # If Q'(y) != 0, it's a simple root (multiplicity 1, odd), so sign changes.
    # If Q'(y) == 0, we need to check higher derivatives or use the epsilon method.
    
    # Let's use the epsilon method for robustness against multiplicity > 1.
    # Evaluate Q(y) at y*(1-delta) and y*(1+delta).
    # If Q(y-delta) * Q(y+delta) < 0, sign changes.
    
    target_R = None
    for R in real_pos_roots:
        # Check sign change
        delta = 1e-6 * max(1.0, abs(R))
        y1 = R - delta
        y2 = R + delta
        
        # Ensure y1 is positive if possible, though Q is defined everywhere.
        # Q(y) = polyval(Q_coeffs, y)
        q1 = np.polyval(Q_coeffs, y1)
        q2 = np.polyval(Q_coeffs, y2)
        
        if q1 * q2 < 0:
            target_R = R
            break # Found the largest one with sign change
            
    if target_R is None:
        # If no sign-changing positive root, the "largest real root at which q changes sign" 
        # might be interpreted as non-existent or we should look further?
        # The problem states "Let r be the largest real root of q at which q changes sign."
        # If none exist, the value is undefined? Or infinite?
        # In the context of the bound, if no such root exists, the bound might be infinite or the configuration is invalid.
        # We return a large value to penalize this.
        return 1e10
        
    r_val = np.sqrt(target_R)
    value = (r_val ** 2) / (2 * np.pi)
    return value

def construct():
    # Seed
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_val = compute_value(seed[0], seed[1], seed[2])
    
    # Global Optimization using Differential Evolution
    # Bounds for c0, c1, c2
    # Based on seed, let's try a box around it, or a wider box.
    # c0 ~ -1.13, c1 ~ 0.04, c2 ~ 0.0003
    # Let's try bounds:
    # c0: [-5, 5]
    # c1: [-1, 1]
    # c2: [-0.1, 0.1]
    
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    def objective(coeffs):
        return compute_value(coeffs[0], coeffs[1], coeffs[2])
        
    try:
        # Differential Evolution is good for global search
        # maxiter=100, popsize=10, tol=1e-8
        # This might take some time, but should be within 180s for 3 variables.
        res_de = differential_evolution(objective, bounds, maxiter=200, popsize=15, tol=1e-8, seed=42)
        
        best_x = res_de.x
        best_val = res_de.fun
        
        # Refine with Nelder-Mead
        res_nm = minimize(objective, best_x, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500})
        
        if res_nm.fun < best_val:
            best_x = res_nm.x
            best_val = res_nm.fun
            
        # Compare with seed
        if best_val < seed_val:
            return best_x
        else:
            return seed
            
    except Exception as e:
        # Fallback to seed if optimization fails
        return seed
# EVOLVE-BLOCK-END
