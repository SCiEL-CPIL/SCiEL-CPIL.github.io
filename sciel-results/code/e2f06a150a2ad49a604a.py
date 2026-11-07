import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite # Not directly useful for arbitrary linear combos, but good for reference
import time
import sys

# Pre-compute Hermite polynomial coefficients for H0, H4, H8, H12
# Physicists' Hermite polynomials: H_n(x)
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 10752x^4 - 10752x^2 + 1344
# H_12 = 4096x^12 - 73728x^10 + 530848x^8 - 1515520x^6 + 1658880x^4 - 619520x^2 + 42240

# Coefficients in descending order of power
H0_coeffs = np.array([1.0])
H4_coeffs = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
H8_coeffs = np.array([256.0, 0.0, -3072.0, 0.0, 10752.0, 0.0, -10752.0, 0.0, 1344.0])
H12_coeffs = np.array([4096.0, 0.0, -73728.0, 0.0, 530848.0, 0.0, -1515520.0, 0.0, 1658880.0, 0.0, -619520.0, 0.0, 42240.0])

# H12(0) is the constant term
H12_0 = H12_coeffs[-1]
H0_0 = H0_coeffs[-1]
H4_0 = H4_coeffs[-1]
H8_0 = H8_coeffs[-1]

def compute_value(c0, c1, c2):
    """
    Computes the objective value for given coefficients c0, c1, c2.
    Returns the value r^2 / (2 * pi). Lower is better.
    Returns a large penalty if invalid.
    """
    # Calculate c3 such that g(0) = 0
    # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Construct g(x) coefficients
    # Pad arrays to same length (13 coefficients for degree 12)
    len_g = 13
    g_coeffs = np.zeros(len_g)
    
    # Add c0 * H0
    # H0 is degree 0, so it aligns with the last element
    g_coeffs[-1] += c0 * H0_0
    
    # Add c1 * H4
    # H4 is degree 4, so it aligns with the last 5 elements
    g_coeffs[-5:] += c1 * H4_coeffs
    
    # Add c2 * H8
    # H8 is degree 8, so it aligns with the last 9 elements
    g_coeffs[-9:] += c2 * H8_coeffs
    
    # Add c3 * H12
    g_coeffs += c3 * H12_coeffs
    
    # Check leading coefficient
    # If g(x) -> -inf as x -> inf, leading coeff is negative. Flip sign.
    if g_coeffs[0] < 0:
        g_coeffs = -g_coeffs
    elif g_coeffs[0] == 0:
        # Degenerate case, unlikely but handle
        # Find first non-zero
        for i in range(len_g):
            if g_coeffs[i] != 0:
                if g_coeffs[i] < 0:
                    g_coeffs = -g_coeffs
                break
        else:
            return 1e100 # Zero polynomial
    
    # q(x) = g(x) / x^2
    # This is valid if g(0)=0 and g'(0)=0?
    # H0(0)=1, H4(0)=12, H8(0)=1344, H12(0)=42240.
    # g(0)=0 is enforced.
    # For division by x^2 to be exact, we need g(x) to be divisible by x^2.
    # This requires the constant term and the x^1 term to be zero.
    # Constant term is g_coeffs[-1] which is 0 by construction.
    # x^1 term is g_coeffs[-2].
    # Let's check if g_coeffs[-2] is close to 0.
    # H0 has no x term. H4 has no x term. H8 has no x term. H12 has no x term.
    # All Hermite polynomials H_n have only powers of x with the same parity as n.
    # H0 (even), H4 (even), H8 (even), H12 (even).
    # So g(x) only has even powers. Thus g(x) is always divisible by x^2 if g(0)=0?
    # No, if g(0)=0, the constant term is 0. The next term is x^2.
    # So yes, division by x^2 is exact.
    
    if abs(g_coeffs[-2]) > 1e-9:
        # Should not happen due to parity, but just in case
        return 1e100
        
    q_coeffs = g_coeffs[:-2] # Remove last two (x^0 and x^1)
    
    # Remove leading zeros to reduce degree
    while len(q_coeffs) > 1 and q_coeffs[0] == 0:
        q_coeffs = q_coeffs[1:]
        
    if len(q_coeffs) == 0:
        return 1e100
        
    # Find roots of q(x)
    roots = np.roots(q_coeffs)
    
    # Filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    
    if len(real_roots) == 0:
        return 1e100
        
    # For each real root, check if sign changes.
    # Since q(x) is a polynomial, sign changes at simple roots.
    # We can check the sign of q(x) slightly before and after the root.
    # Or simply, if the root is simple, it's a sign change.
    # Multiplicity can be checked by derivative, but np.roots gives distinct roots mostly.
    # A safer check: evaluate q at root - eps and root + eps.
    # However, for the "largest" root, we just need to identify valid ones.
    
    valid_roots = []
    for r in real_roots:
        # Check sign change
        # Evaluate q at r - 1e-6 and r + 1e-6
        # Note: if r is very large, this might be unstable.
        # But r is expected to be around 2-3.
        
        # Alternative: The problem states "largest real root of q at which q changes sign".
        # For a polynomial with real coefficients, non-real roots come in conjugate pairs.
        # Real roots of odd multiplicity involve sign changes.
        # We can assume simple roots for the optimal case.
        
        # Let's just take the largest real root. 
        # Does it always change sign?
        # q(x) -> +inf as x -> +inf (since we ensured leading coeff > 0).
        # If the largest real root is simple, it must change sign from + to - or - to +?
        # Since it's the largest, to the right of it q(x) is positive.
        # So it must change from negative to positive at the root?
        # Or if it's a double root, it touches 0 and stays positive (no sign change).
        # We need a sign change.
        
        # Let's verify sign change for the largest few roots.
        pass
    
    # More robust: sort real roots, check from largest down.
    real_roots.sort(reverse=True)
    
    for r in real_roots:
        # Check sign change at r
        # q(r) should be close to 0
        # Check q(r - delta) and q(r + delta)
        delta = 1e-5
        # Ensure we don't step over another root
        # This is heuristic.
        
        # Evaluate polynomial at points
        # Using np.polyval
        q_left = np.polyval(q_coeffs, r - delta)
        q_right = np.polyval(q_coeffs, r + delta)
        
        if q_left * q_right < 0:
            # Sign change detected
            return (r * r) / (2 * np.pi)
            
    # If no sign-changing root found
    return 1e100

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 # 150s limit to be safe
    
    # Initial best from seed
    c_seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = compute_value(c_seed[0], c_seed[1], c_seed[2])
    best_c = c_seed.copy()
    
    # Define a set of starting points for Nelder-Mead
    # 1. The seed
    # 2. Perturbations of the seed
    # 3. Random points in a reasonable box
    
    starts = [c_seed]
    
    # Small perturbations
    for scale in [0.1, 0.01, 0.001]:
        for i in range(3):
            noise = np.random.uniform(-scale, scale, 3)
            starts.append(c_seed + noise)
            
    # Broader random search
    # Coefficients are likely O(1) or smaller.
    for i in range(10):
        if time.time() > deadline:
            break
        r = np.random.uniform(-1.0, 1.0, 3)
        starts.append(r)
        
    # Run Nelder-Mead for each start
    options = {
        'maxiter': 200,
        'fatol': 1e-12,
        'xatol': 1e-12,
        'adaptive': True
    }
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
            
        def objective(c):
            return compute_value(c[0], c[1], c[2])
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options=options)
            val = res.fun
            if val < best_val:
                best_val = val
                best_c = res.x
        except Exception:
            pass
            
    # If we have time left, do a few more local polishes around the best found
    if time.time() < deadline - 10:
        for i in range(5):
            noise = np.random.uniform(-0.01, 0.01, 3)
            x0 = best_c + noise
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options=options)
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_c = res.x
            except Exception:
                pass

    return best_c

# Ensure numpy is available
# The prompt says NumPy is available.

if __name__ == "__main__":
    # Simple self-test
    c = construct()
    print(f"Best coefficients: {c}")
    print(f"Best value: {compute_value(c[0], c[1], c[2])}")
