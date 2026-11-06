import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0 # 110s safety margin out of 120s

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Filter out any non-finite values just in case
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)

    # Scale x to [-1, 1] for numerical stability in Legendre fit
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Degenerate case: all x are the same
        # Fit a constant (degree 0)
        c0 = np.mean(y)
        return np.array([c0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    x_scaled = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)
    
    # Fit Legendre polynomial of degree 5
    # legfit returns coefficients in ascending order of Legendre polynomials
    leg_coeffs = np.polynomial.legendre.legfit(x_scaled, y, 5)
    
    # Convert Legendre coefficients to standard monomial coefficients
    # We need to map P_k( (x - mid) / half ) to c_i * x^i
    # This is a linear transformation.
    # Let u = (x - mid) / half. x = half * u + mid.
    # P_k(u) is a polynomial in u of degree k.
    # We can use numpy's conversion utilities.
    
    # First, convert Legendre coeffs to monomial coeffs in the scaled variable u
    # leg2poly converts Legendre coefficients to monomial coefficients
    # The result will be coefficients for powers of u: sum a_i * u^i
    monomial_u = np.polynomial.legendre.leg2poly(leg_coeffs)
    
    # Now we have y ≈ sum_{i=0}^5 a_i * u^i
    # u = (x - mid) / half
    # We need to expand sum a_i * ((x - mid) / half)^i into sum c_j * x^j
    # This is a change of basis from u to x.
    # We can construct the transformation matrix or use polyfromroots/polyfit tricks, 
    # but direct expansion is safer and fast for degree 5.
    
    # Let's build the transformation matrix T such that monomial_x = T @ monomial_u
    # u = (1/half) * x - (mid/half)
    # u^0 = 1
    # u^1 = (1/half)*x - mid/half
    # u^2 = (1/half)^2 * x^2 - 2*(1/half)*(mid/half)*x + (mid/half)^2
    # ...
    
    # It is easier to use numpy's poly1d or manual expansion for small degree.
    # Let's use a simple loop to accumulate coefficients.
    
    c_x = np.zeros(6)
    
    # u = A * x + B
    A = 1.0 / (x_range / 2.0)
    B = -(x_min + x_max) / 2.0 / (x_range / 2.0)
    
    # We want to compute sum_{i=0}^5 a_i * (A*x + B)^i
    # We can iteratively multiply polynomials.
    # Current polynomial in x: P(x)
    # Start with P(x) = 0
    # For each i, add a_i * (A*x+B)^i
    
    # Let's compute powers of (A*x+B)
    # power_0 = 1
    # power_1 = A*x + B
    # power_k = power_{k-1} * (A*x+B)
    
    powers = []
    current_power = np.array([1.0]) # constant 1
    for i in range(6):
        powers.append(current_power.copy())
        if i < 5:
            # Multiply current_power (which is coeff for u^i) by (A*x + B)
            # (c0 + c1*x + ... + cn*x^n) * (B + A*x)
            # new_c0 = c0*B
            # new_ck = c_{k-1}*A + c_k*B
            # new_c_{n+1} = c_n*A
            new_len = len(current_power) + 1
            new_power = np.zeros(new_len)
            new_power[0] = current_power[0] * B
            for k in range(1, len(current_power)):
                new_power[k] = current_power[k-1] * A + current_power[k] * B
            new_power[-1] = current_power[-1] * A
            current_power = new_power
            
    # Now sum a_i * powers[i]
    for i in range(6):
        p = powers[i]
        # Pad p to length 6
        padded_p = np.zeros(6)
        padded_p[:len(p)] = p
        c_x += monomial_u[i] * padded_p
        
    # Check time
    if time.time() > deadline:
        # Fallback to linear if somehow slow (unlikely)
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
        
    return c_x
