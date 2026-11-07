"""Rewritten program: local search around the seed to minimize the value."""
# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import sys

# We'll implement the evaluation locally using sympy for exact arithmetic,
# but fall back to a fast numerical approximation if sympy is too slow.
# However, the task says "The evaluator computes exactly this value... with exact rational arithmetic".
# We need to replicate the evaluator's logic to guide our search.
# The evaluator uses sympy.polys.orthopolys.hermite_poly.

# Let's try to import sympy. If it's available, we use it for exact evaluation.
# If not, we approximate. Given the 180s limit, we must be careful.
# Actually, the prompt says "Python standard library, NumPy and SciPy are available."
# It does *not* explicitly say Sympy is available in the *candidate* environment,
# but the evaluator uses it.
# Wait, usually in these tasks, if the evaluator uses a library, it might be available.
# Let's check if sympy is importable. If not, we'll have to approximate the value.
# The value is r^2 / (2*pi), where r is the largest real root of q(x) = g(x)/x^2 that changes sign.
# q(x) is a polynomial of degree 10 (since g is degree 12, divided by x^2).
# We can compute the coefficients of g(x) and then q(x) using NumPy/SciPy.

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

# Precompute Hermite polynomials coefficients using NumPy/SciPy or hardcoded?
# Hermite polynomials H_n(x) (physicists) satisfy:
# H_0(x) = 1
# H_1(x) = 2x
# H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
# We need H_0, H_4, H_8, H_12.
# We can generate them as numpy arrays of coefficients (highest degree first, or lowest?).
# Let's use lowest degree first for easier convolution/division.

def hermite_coeffs(n):
    """Return coefficients of H_n(x) as a list [c0, c1, ..., cn] where c_i is coeff of x^i."""
    if n == 0:
        return [1]
    if n == 1:
        return [0, 2]
    
    H_nm2 = [1]  # H_0
    H_nm1 = [0, 2] # H_1
    
    for k in range(2, n + 1):
        # H_k = 2x H_{k-1} - 2(k-1) H_{k-2}
        # 2x H_{k-1}: shift H_{k-1} by 1 and multiply by 2
        term1 = [0] + [2 * c for c in H_nm1]
        # -2(k-1) H_{k-2}: multiply H_{k-2} by -2(k-1)
        term2 = [-2 * (k - 1) * c for c in H_nm2]
        
        # Ensure same length
        max_len = max(len(term1), len(term2))
        while len(term1) < max_len:
            term1.append(0)
        while len(term2) < max_len:
            term2.append(0)
            
        H_n = [t1 + t2 for t1, t2 in zip(term1, term2)]
        
        H_nm2 = H_nm1
        H_nm1 = H_n
        
    return H_nm1

# Precompute coefficients for H0, H4, H8, H12
H0_coeffs = hermite_coeffs(0)  # [1]
H4_coeffs = hermite_coeffs(4)  # [0, 0, 0, 0, 16] ? Let's check.
# H_2 = 2x(2x) - 2(1)(1) = 4x^2 - 2
# H_3 = 2x(4x^2-2) - 2(2)(2x) = 8x^3 - 4x - 8x = 8x^3 - 12x
# H_4 = 2x(8x^3-12x) - 2(3)(4x^2-2) = 16x^4 - 24x^2 - 24x^2 + 12 = 16x^4 - 48x^2 + 12
# So H4_coeffs should be [12, 0, -48, 0, 16]
# Let's verify with the function.

H8_coeffs = hermite_coeffs(8)
H12_coeffs = hermite_coeffs(12)

def evaluate_value(c0, c1, c2):
    """
    Compute the value for given coefficients c0, c1, c2.
    Returns the value (float) or None if invalid.
    """
    # Convert to Fraction for exact arithmetic if possible, or use float for speed?
    # The prompt says "exact rational arithmetic on your floats".
    # This implies the evaluator converts our floats to Fractions.
    # We can simulate this by using Fraction(c0) etc.
    
    try:
        c0_f = Fraction(c0)
        c1_f = Fraction(c1)
        c2_f = Fraction(c2)
    except:
        return None

    # Build g(x) = c0 H0 + c1 H4 + c2 H8 + c3 H12
    # We need coefficients of g(x).
    # g(x) is a polynomial of degree 12.
    # Coefficients array of length 13 (index 0 to 12).
    
    # Helper to add scaled polynomial
    def add_poly(base, coeffs, scale):
        # base is list of Fractions
        # coeffs is list of ints
        # scale is Fraction
        for i, c in enumerate(coeffs):
            base[i] += scale * c
        return base

    g_coeffs = [Fraction(0) for _ in range(13)]
    add_poly(g_coeffs, H0_coeffs, c0_f)
    add_poly(g_coeffs, H4_coeffs, c1_f)
    add_poly(g_coeffs, H8_coeffs, c2_f)
    
    # Calculate c3
    # g(0) = c0 H0(0) + c1 H4(0) + c2 H8(0) + c3 H12(0) = 0
    # H_n(0) is the constant term (index 0)
    H0_0 = Fraction(H0_coeffs[0])
    H4_0 = Fraction(H4_coeffs[0])
    H8_0 = Fraction(H8_coeffs[0])
    H12_0 = Fraction(H12_coeffs[0])
    
    if H12_0 == 0:
        return None # Should not happen for H12
    
    c3_f = -(c0_f * H0_0 + c1_f * H4_0 + c2_f * H8_0) / H12_0
    
    add_poly(g_coeffs, H12_coeffs, c3_f)
    
    # Check if g tends to a negative limit at infinity.
    # Leading coefficient is g_coeffs[12].
    if g_coeffs[12] < 0:
        # Replace g by -g
        g_coeffs = [-c for c in g_coeffs]
    
    # q(x) = g(x) / x^2
    # g(x) must be divisible by x^2.
    # Check g_coeffs[0] == 0 and g_coeffs[1] == 0.
    if g_coeffs[0] != 0 or g_coeffs[1] != 0:
        return None # Should not happen if c3 is calculated correctly?
        # Actually, g(0)=0 ensures g_coeffs[0]=0.
        # Does g'(0)=0?
        # g'(x) = c0 H0' + c1 H4' + c2 H8' + c3 H12'.
        # H_n'(0) = 0 for even n. H0, H4, H8, H12 are all even polynomials.
        # So their derivatives at 0 are 0.
        # So g'(0) = 0. Thus g_coeffs[1] should be 0.
    
    # q_coeffs has degree 10.
    # q_coeffs[i] = g_coeffs[i+2]
    q_coeffs = [Fraction(0) for _ in range(11)]
    for i in range(11):
        q_coeffs[i] = g_coeffs[i+2]
    
    # Find the largest real root of q(x) at which q changes sign.
    # We can convert to float for root finding using NumPy/SciPy.
    # q_coeffs are Fractions. Convert to float.
    q_floats = [float(c) for c in q_coeffs]
    
    # np.roots expects highest degree first.
    # q_coeffs is lowest degree first.
    q_reversed = q_floats[::-1]
    
    try:
        roots = np.roots(q_reversed)
    except:
        return None
    
    # Filter for real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    # Check sign change at each real root.
    # We need to evaluate q(x) at points slightly to the left and right of the root.
    # Or check the multiplicity. If multiplicity is odd, sign changes.
    # Since we are using float roots, multiplicity detection is tricky.
    # Better to evaluate q(x) at r - eps and r + eps.
    
    def q_val(x):
        # Evaluate polynomial q(x) using Horner's method
        val = 0
        for c in q_coeffs[::-1]:
            val = val * x + c
        return val
    
    # We want the LARGEST real root with sign change.
    # Sort roots in descending order.
    real_roots.sort(reverse=True)
    
    r_best = None
    for r in real_roots:
        eps = 1e-6
        q_left = q_val(r - eps)
        q_right = q_val(r + eps)
        
        # Check if sign changes
        # q_left and q_right should have opposite signs
        if q_left * q_right < 0:
            r_best = r
            break
    
    if r_best is None:
        return None
    
    # value = r^2 / (2 * pi)
    value = (r_best ** 2) / (2 * np.pi)
    
    return value

def construct():
    # Start with the seed
    best_c = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = evaluate_value(best_c[0], best_c[1], best_c[2])
    
    if best_val is None:
        return best_c
    
    # Local search
    # Use a simple gradient-free method like Nelder-Mead or coordinate descent.
    # Since we have only 3 parameters, Nelder-Mead is a good choice.
    # We can implement a simple version or use scipy.optimize.minimize if available.
    
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except:
        has_scipy = False
    
    start_time = time.time()
    deadline = 170.0  # Safety margin
    
    if has_scipy:
        def objective(x):
            val = evaluate_value(x[0], x[1], x[2])
            if val is None:
                return 1e10  # Large penalty
            return val
        
        # Use Nelder-Mead
        # Initial simplex
        x0 = best_c
        # Perturb slightly to create simplex
        # Or let scipy handle it
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12})
        
        if time.time() < deadline:
            # Check if result is better
            val_new = evaluate_value(res.x[0], res.x[1], res.x[2])
            if val_new is not None and val_new < best_val:
                best_c = res.x
                best_val = val_new
    
    # If scipy is not available or didn't improve, do a manual local search
    # Simple coordinate descent with adaptive step size
    
    step = 1e-4
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(3):
            for direction in [1, -1]:
                c_new = best_c.copy()
                c_new[i] += direction * step
                val_new = evaluate_value(c_new[0], c_new[1], c_new[2])
                if val_new is not None and val_new < best_val:
                    best_c = c_new
                    best_val = val_new
                    improved = True
            # If we moved, shrink step? Or keep?
            # If we didn't move, shrink step for this dimension
        if not improved:
            step *= 0.5
            if step < 1e-12:
                break
    
    return best_c
# EVOLVE-BLOCK-END
