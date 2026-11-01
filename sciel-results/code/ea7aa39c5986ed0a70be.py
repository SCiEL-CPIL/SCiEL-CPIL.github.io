import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Hermite polynomials coefficients (physicist's convention H_n(x))
# H_0 = 1
# H_4 = x^4 - 6x^2 + 3
# H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12 = x^12 - 84x^10 + 2100x^8 - 25200x^6 + 138600x^4 - 332640x^2 + 10395

# Coefficients for H_0, H_4, H_8, H_12 in descending power of x
# H_0: [1] (degree 0)
# H_4: [1, 0, -6, 0, 3] (degree 4)
# H_8: [1, 0, -28, 0, 210, 0, -420, 0, 105] (degree 8)
# H_12: [1, 0, -84, 0, 2100, 0, -25200, 0, 138600, 0, -332640, 0, 10395] (degree 12)

# We represent polynomials as arrays of coefficients from highest degree to lowest.
# To make addition and multiplication easier, let's use a dictionary or fixed length array.
# Max degree is 12. P(x)^2 has degree 24.
# Let's use numpy arrays of length 25 (indices 0 to 24 for x^24 to x^0).

def get_poly_coeffs(coeffs_list, degrees):
    """
    coeffs_list: list of coefficients for each Hermite polynomial
    degrees: list of degrees [0, 4, 8, 12]
    Returns: numpy array of length 25 (coeffs for x^24 ... x^0)
    """
    poly = np.zeros(25)
    for c, deg in zip(coeffs_list, degrees):
        if c == 0: continue
        # Get the specific Hermite polynomial coefficients
        if deg == 0:
            h = np.array([1.0])
        elif deg == 4:
            h = np.array([1.0, 0.0, -6.0, 0.0, 3.0])
        elif deg == 8:
            h = np.array([1.0, 0.0, -28.0, 0.0, 210.0, 0.0, -420.0, 0.0, 105.0])
        elif deg == 12:
            h = np.array([1.0, 0.0, -84.0, 0.0, 2100.0, 0.0, -25200.0, 0.0, 138600.0, 0.0, -332640.0, 0.0, 10395.0])
        else:
            continue
            
        # Pad h to length 13 (degree 12)
        h_pad = np.zeros(13)
        h_pad[:len(h)] = h
        
        # Shift to correct position in the 25-length array?
        # No, we are summing polynomials.
        # P(x) = sum c_i H_i(x).
        # We can just add c_i * h_i to the sum.
        # Let's maintain a sum array of length 13 for P(x) first.
        pass
        
    # Let's refactor to be cleaner.
    # P(x) is degree 12.
    p_coeffs = np.zeros(13) # indices 0..12 correspond to x^12 .. x^0
    for c, deg in zip(coeffs_list, degrees):
        if deg == 0:
            p_coeffs[12] += c * 1.0
        elif deg == 4:
            p_coeffs[12] += c * 1.0
            p_coeffs[10] += c * (-6.0)
            p_coeffs[8]  += c * 3.0
        elif deg == 8:
            p_coeffs[12] += c * 1.0
            p_coeffs[10] += c * (-28.0)
            p_coeffs[8]  += c * 210.0
            p_coeffs[6]  += c * (-420.0)
            p_coeffs[4]  += c * 105.0
        elif deg == 12:
            p_coeffs[12] += c * 1.0
            p_coeffs[10] += c * (-84.0)
            p_coeffs[8]  += c * 2100.0
            p_coeffs[6]  += c * (-25200.0)
            p_coeffs[4]  += c * 138600.0
            p_coeffs[2]  += c * (-332640.0)
            p_coeffs[0]  += c * 10395.0
            
    return p_coeffs

def compute_moments(p_coeffs):
    """
    p_coeffs: array of length 13, p_coeffs[i] is coeff of x^(12-i).
    Computes I_0 = int P(x)^2 e^-x^2 dx and I_2 = int x^2 P(x)^2 e^-x^2 dx.
    """
    # Square the polynomial
    # p_coeffs has degree 12. p^2 has degree 24.
    # Use np.convolve
    p_squared = np.convolve(p_coeffs, p_coeffs)
    # p_squared length is 25.
    # p_squared[0] is coeff of x^24.
    # p_squared[24] is coeff of x^0.
    
    # Moments of e^-x^2:
    # M_k = int x^k e^-x^2 dx
    # M_k = 0 for odd k.
    # M_{2m} = (2m-1)!! * sqrt(pi) / 2^m
    # Let's precompute M_k for k=0..24.
    
    M = np.zeros(25)
    M[0] = np.sqrt(np.pi)
    # Recurrence: M_{2m} = (2m-1) * M_{2m-2} / 2
    # M_0 = sqrt(pi)
    # M_2 = 1 * M_0 / 2
    # M_4 = 3 * M_2 / 2
    # M_6 = 5 * M_4 / 2
    # ...
    # M_{2m} = (2m)! / (4^m m!) * sqrt(pi)
    
    for k in range(2, 25, 2):
        m = k // 2
        # M_k = (2m-1) * M_{k-2} / 2
        M[k] = (k - 1) * M[k - 2] / 2.0
        
    # I_0 = sum p_squared[i] * M[24 - 2*i] ?
    # p_squared[i] is coeff of x^(24 - i).
    # Term is p_squared[i] * x^(24-i) * e^-x^2.
    # Integral is p_squared[i] * M[24-i].
    # Since M is 0 for odd, we can just sum all.
    
    I_0 = np.dot(p_squared, M)
    
    # I_2 = int x^2 P(x)^2 e^-x^2 dx
    # Term is p_squared[i] * x^(24-i) * x^2 * e^-x^2 = p_squared[i] * x^(26-i) * e^-x^2
    # Integral is p_squared[i] * M[26-i].
    # We need M up to 26.
    M_ext = np.zeros(27)
    M_ext[:25] = M
    M_ext[25] = 0
    M_ext[26] = 25 * M_ext[24] / 2.0
    
    I_2 = np.dot(p_squared, M_ext[2:]) # M[2], M[4]... M[26]
    # Wait.
    # p_squared[0] (x^24) -> M[26]
    # p_squared[1] (x^23) -> M[25] (0)
    # ...
    # p_squared[24] (x^0) -> M[2]
    
    # So I_2 = sum p_squared[i] * M[26-i]
    # Let's create M_shifted where M_shifted[i] = M[26-i]
    M_shifted = np.zeros(25)
    for i in range(25):
        idx = 26 - i
        if 0 <= idx < 27:
            M_shifted[i] = M_ext[idx]
            
    I_2 = np.dot(p_squared, M_shifted)
    
    return I_0, I_2

def objective(c):
    c0, c4, c8 = c
    # Calculate c12
    # P(0) = c0*1 + c4*3 + c8*105 + c12*945 = 0
    # c12 = -(c0 + 3*c4 + 105*c8) / 945
    c12 = -(c0 + 3*c4 + 105*c8) / 945.0
    
    coeffs = [c0, c4, c8, c12]
    degrees = [0, 4, 8, 12]
    
    p_coeffs = get_poly_coeffs(coeffs, degrees)
    
    # Check if P is zero
    if np.allclose(p_coeffs, 0):
        return 1e100
        
    I_0, I_2 = compute_moments(p_coeffs)
    
    if I_0 < 1e-10:
        return 1e100
        
    return I_2 / I_0

def construct():
    start_time = time.time()
    deadline = start_time + 170  # 10s safety margin
    
    # Initial guess from current program
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_val = objective(x0)
    best_x = x0
    
    # Try a few random restarts and local optimizations
    np.random.seed(42)
    
    # Grid of initial points
    # The coefficients might be small.
    # Let's try some structured points.
    restarts = [x0]
    # Add some random perturbations
    for _ in range(10):
        x_rand = x0 + np.random.normal(0, 0.1, 3)
        restarts.append(x_rand)
        
    # Also try scaling variations
    for scale in [0.1, 10, 0.01, 100]:
        x_scaled = x0 * scale
        restarts.append(x_scaled)
        
    for x_init in restarts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x_init, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000})
            if res.success:
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
        except Exception as e:
            pass
            
    # Return the coefficients c0, c4, c8
    # The function should return the 3 coefficients.
    # The task says: "construct() -> three coefficients ... the degree-12 weight is fixed"
    # So we return best_x
    
    return np.array(best_x)

# EVOLVE-BLOCK-END
