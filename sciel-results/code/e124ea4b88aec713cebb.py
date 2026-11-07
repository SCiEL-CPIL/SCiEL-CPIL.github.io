"""Rewritten program for the suite task; searches for better coefficients."""
import numpy as np
import time
import sys
from fractions import Fraction

# We will use sympy for exact arithmetic if available, otherwise fall back to high-precision floats.
# The task says "exact rational arithmetic on your floats". This implies the evaluator converts our floats to rationals.
# So we should output floats that are close to good rational numbers.
# However, to find the best coefficients, we can optimize in the continuous domain.

# Let's check if sympy is available for better root finding or exactness.
import sympy as sp

# Precompute Hermite polynomials H0, H4, H8, H12
# H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 1792x^6 + 3360x^4 - 1344x^2 + 168
# H_12 = 4096x^12 - 55296x^10 + 308736x^8 - 564480x^6 + 304128x^4 - 42240x^2 + 1032

# Let's verify these coefficients using sympy to be sure.
x = sp.Symbol('x')
H0 = sp.hermite(0, x)
H4 = sp.hermite(4, x)
H8 = sp.hermite(8, x)
H12 = sp.hermite(12, x)

# Coefficients for g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
# H0(0) = 1
# H4(0) = 12
# H8(0) = 168
# H12(0) = 1032

# c3 = -(c0*1 + c1*12 + c2*168) / 1032

# q(x) = g(x) / x^2
# We need q(x) to have a largest real root r where it changes sign.
# value = r^2 / (2*pi)

# We want to minimize value, so we want to minimize r^2.
# This means we want to push the largest sign-changing root of q(x) as close to 0 as possible.
# But q(x) must be well-behaved.

# Let's analyze the behavior of q(x) for the seed.
# c0 = -1.13, c1 = 0.04, c2 = 1/3240 ≈ 0.0003086
# c3 = -(-1.13 + 12*0.04 + 168*(1/3240)) / 1032
# c3 = -(-1.13 + 0.48 + 0.052) / 1032 = -(-0.598) / 1032 ≈ 0.00058

# Let's use a numerical optimization approach.
# We define a function f(c0, c1, c2) that computes the value.
# We need to be careful with the "largest real root where q changes sign".

def compute_value(c0, c1, c2):
    # Compute c3
    c3 = -(c0 * 1 + c1 * 12 + c2 * 168) / 1032.0
    
    # Construct g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # We work with q(x) = g(x) / x^2
    # g(x) is a polynomial of degree 12.
    # q(x) is a polynomial of degree 10.
    
    # Let's get the coefficients of g(x) in standard basis [1, x, x^2, ..., x^12]
    # H0 = 1
    # H4 = 12 - 48x^2 + 16x^4
    # H8 = 168 - 1344x^2 + 3360x^4 - 1792x^6 + 256x^8
    # H12 = 1032 - 42240x^2 + 304128x^4 - 564480x^6 + 308736x^8 - 55296x^10 + 4096x^12
    
    # g(x) coefficients:
    # coeff_0 = c0*1 + c1*12 + c2*168 + c3*1032  (should be 0)
    # coeff_2 = c1*(-48) + c2*(-1344) + c3*(-42240)
    # coeff_4 = c1*(16) + c2*(3360) + c3*(304128)
    # coeff_6 = c2*(-1792) + c3*(-564480)
    # coeff_8 = c2*(256) + c3*(308736)
    # coeff_10 = c3*(-55296)
    # coeff_12 = c3*(4096)
    
    # q(x) = g(x)/x^2 = coeff_2 + coeff_4*x^2 + coeff_6*x^4 + coeff_8*x^6 + coeff_10*x^8 + coeff_12*x^10
    
    c2_g = c1 * (-48) + c2 * (-1344) + c3 * (-42240)
    c4_g = c1 * (16) + c2 * (3360) + c3 * (304128)
    c6_g = c2 * (-1792) + c3 * (-564480)
    c8_g = c2 * (256) + c3 * (308736)
    c10_g = c3 * (-55296)
    c12_g = c3 * (4096)
    
    # q(x) is a polynomial in x^2. Let y = x^2.
    # P(y) = c2_g + c4_g*y + c6_g*y^2 + c8_g*y^3 + c10_g*y^4 + c12_g*y^5
    # We need the largest real root r of q(x) where q changes sign.
    # This corresponds to the largest positive root y_r of P(y) where P(y) changes sign (i.e., simple root).
    # Then r = sqrt(y_r).
    
    # If P(y) has no positive roots, or the largest positive root is not a sign change, what then?
    # The problem states "Let r be the largest real root of q at which q changes sign."
    # If no such root exists, the behavior is undefined or infinite?
    # Usually, for these bounds, we want q(x) to be positive for large x (so g(x) ~ x^2 q(x) > 0) and have a zero at r.
    # If q(x) is always positive, r might be 0? Or maybe the problem implies such a root exists.
    # Let's assume we want to find the largest positive simple root of P(y).
    
    coeffs_P = [c12_g, c10_g, c8_g, c6_g, c4_g, c2_g]
    
    # Find roots of P(y)
    try:
        roots = np.roots(coeffs_P)
    except Exception:
        return float('inf')
        
    # Filter real positive roots
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-9:
            real_part = root.real
            if real_part > 1e-9: # Must be positive
                # Check if it's a sign change.
                # For a simple root, the sign changes.
                # We can evaluate P(y) at y - eps and y + eps.
                y_test = real_part
                # Derivative P'(y)
                # P'(y) = c4_g + 2*c6_g*y + 3*c8_g*y^2 + 4*c10_g*y^3 + 5*c12_g*y^4
                # If P'(y) != 0, it's a simple root.
                deriv = c4_g + 2*c6_g*y_test + 3*c8_g*y_test**2 + 4*c10_g*y_test**3 + 5*c12_g*y_test**4
                if abs(deriv) > 1e-9:
                    real_roots.append(real_part)
                    
    if not real_roots:
        # If no sign-changing positive root, what is the value?
        # If q(x) > 0 for all x, then the "largest root" is not defined in the same way.
        # However, if q(x) is positive, then g(x) is positive for large x.
        # The condition "g tends to a negative limit at infinity, g is replaced by -g" suggests we care about the sign at infinity.
        # If q(x) > 0 for all x, then r is effectively 0? Or maybe the bound is 0?
        # Let's check the seed.
        # Seed value is 1.0.
        # If we return 0, score is inf?
        # Let's assume if no root, value is 0.
        return 0.0
        
    y_r = max(real_roots)
    r = np.sqrt(y_r)
    value = r**2 / (2 * np.pi)
    return value

# Optimization
# We want to minimize compute_value(c0, c1, c2).
# We can use Nelder-Mead or similar from scipy.optimize.
# But we have a time limit.
# Let's do a simple local search around the seed.

def construct():
    start_time = time.time()
    deadline = start_time + 200  # 10s safety margin
    
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_c = seed
    best_val = compute_value(seed[0], seed[1], seed[2])
    
    # Use scipy.optimize.minimize
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback to simple random search
        for i in range(1000):
            if time.time() > deadline:
                break
            c = seed + np.random.normal(0, 0.01, 3)
            val = compute_value(c[0], c[1], c[2])
            if val < best_val:
                best_val = val
                best_c = c
        return best_c
        
    # Define objective
    def obj(c):
        return compute_value(c[0], c[1], c[2])
    
    # Try Nelder-Mead
    # We might need to handle cases where compute_value returns inf or is noisy.
    # Nelder-Mead is robust to some noise.
    
    # Let's run multiple restarts with different initial points.
    # But keep it fast.
    
    # First, try refining the seed.
    res = minimize(obj, seed, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000, 'maxfev': 1000})
    if time.time() < deadline:
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    
    # Try some other points
    # The space of coefficients is 3D.
    # Let's sample a few points.
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline - 10:
            break
        # Random point in a box around seed
        c = seed + np.random.uniform(-0.5, 0.5, 3)
        res = minimize(obj, c, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 500})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
            
    return best_c

# Wait, the task says "A float rounds to a nearby rational".
# This suggests that the exact value depends on the rational approximation of our float.
# This might create a "jagged" landscape.
# However, for the purpose of finding a good candidate, continuous optimization should get us close.
# The evaluator will then compute the exact value for that specific rational.
# If the continuous minimum is a local minimum, the rational nearby should also be good.

# Let's double check the logic of "largest real root where q changes sign".
# If q(x) has multiple roots, we take the largest one.
# If q(x) is positive for all x, there are no such roots.
# In that case, is the value 0?
# If value is 0, score is inf. That would be a perfect score.
# Can we make q(x) > 0 for all x?
# q(x) = c12_g x^10 + c10_g x^8 + c8_g x^6 + c6_g x^4 + c4_g x^2 + c2_g
# For q(x) > 0 for all x, we need the polynomial in y=x^2 to be positive for all y>=0.
# This requires the polynomial to be a sum of squares or similar.
# Let's see if we can find such coefficients.
# If we can, the value is 0.
# But the seed has a positive value.
# Maybe it's impossible to have q(x) > 0 for all x with g(0)=0 and the specific Hermite basis?
# Or maybe the "largest root" definition implies something else.
# "Let r be the largest real root of q at which q changes sign."
# If no such root exists, the problem statement doesn't specify.
# But usually in these inequalities, if the function doesn't cross zero, the constant is 0 or undefined.
# Given the seed score is 1.0, and value is positive, it implies roots exist.

# Let's refine the search to be more robust.
# Also, ensure we return a numpy array.

def construct():
    start_time = time.time()
    deadline = start_time + 200
    
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_c = seed.copy()
    best_val = compute_value(seed[0], seed[1], seed[2])
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback
        for i in range(5000):
            if time.time() > deadline:
                break
            c = seed + np.random.normal(0, 0.05, 3)
            val = compute_value(c[0], c[1], c[2])
            if val < best_val:
                best_val = val
                best_c = c
        return best_c
        
    def obj(c):
        try:
            return compute_value(c[0], c[1], c[2])
        except Exception:
            return 1e10
            
    # Run Nelder-Mead from seed
    res = minimize(obj, seed, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 2000})
    if time.time() < deadline - 5:
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    
    # Run from a few other points
    # Explore the neighborhood
    for i in range(10):
        if time.time() > deadline - 10:
            break
        # Jitter
        jitter = np.random.normal(0, 0.1, 3)
        c0 = seed + jitter
        res = minimize(obj, c0, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000, 'maxfev': 1000})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
            
    return best_c

# One more check: the problem says "If g tends to a negative limit at infinity, g is replaced by -g."
# g(x) ~ c3 * 4096 * x^12.
# So if c3 < 0, g(x) -> -infinity. Then g is replaced by -g.
# This means the sign of the leading coefficient of g matters.
# If we flip g, the roots of q remain the same, but the sign of q flips.
# "largest real root of q at which q changes sign".
# If q changes sign at r, -q also changes sign at r.
# So the value r^2 is invariant under g -> -g.
# So we don't need to worry about this condition for the value calculation, as long as we identify the roots correctly.
# The condition just ensures a canonical form, probably to avoid ambiguity if "largest root" depended on sign (e.g. if we looked for the first root from 0).
# But "largest real root" is well-defined regardless of sign.

# Let's add a check: what if c3 is 0?
# Then g is degree 8.
# q is degree 6.
# Code handles this naturally.

# What if c3 is very small?
# Numerical issues with roots.
# Using np.roots can be unstable for high degree polynomials.
# But degree 10 is manageable.

# Let's try to improve the initial search points.
# The seed is [-1.13, 0.04, 0.0003].
# Let's try scaling c1 and c2.

# Final check on the return type.
# "Interface: construct() -> three coefficients [c0, c1, c2]."
# "Keep the function names and signatures of the seed program."
# Seed returns np.array.
# So we return np.array.

# Let's ensure the deadline guard is strict.
# 210s limit.
# We use 200s.

# One potential issue: `compute_value` might be slow if called many times.
# `np.roots` is fast for degree 10.
# 2000 iterations * 10ms = 20s.
# 10 restarts * 1000 iterations = 10s.
# Total ~30s. Safe.

# Let's write the final code.

import numpy as np
import time

def compute_value(c0, c1, c2):
    c3 = -(c0 * 1 + c1 * 12 + c2 * 168) / 1032.0
    
    c2_g = c1 * (-48) + c2 * (-1344) + c3 * (-42240)
    c4_g = c1 * (16) + c2 * (3360) + c3 * (304128)
    c6_g = c2 * (-1792) + c3 * (-564480)
    c8_g = c2 * (256) + c3 * (308736)
    c10_g = c3 * (-55296)
    c12_g = c3 * (4096)
    
    coeffs_P = [c12_g, c10_g, c8_g, c6_g, c4_g, c2_g]
    
    try:
        roots = np.roots(coeffs_P)
    except Exception:
        return 1e10
        
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-9:
            real_part = root.real
            if real_part > 1e-9:
                y_test = real_part
                deriv = c4_g + 2*c6_g*y_test + 3*c8_g*y_test**2 + 4*c10_g*y_test**3 + 5*c12_g*y_test**4
                if abs(deriv) > 1e-9:
                    real_roots.append(real_part)
                    
    if not real_roots:
        return 0.0
        
    y_r = max(real_roots)
    r = np.sqrt(y_r)
    value = r**2 / (2 * np.pi)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 200
    
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_c = seed.copy()
    best_val = compute_value(seed[0], seed[1], seed[2])
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback random search
        for i in range(5000):
            if time.time() > deadline:
                break
            c = seed + np.random.normal(0, 0.05, 3)
            val = compute_value(c[0], c[1], c[2])
            if val < best_val:
                best_val = val
                best_c = c
        return best_c
        
    def obj(c):
        try:
            return compute_value(c[0], c[1], c[2])
        except Exception:
            return 1e10
            
    # Primary optimization from seed
    res = minimize(obj, seed, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 2000})
    if time.time() < deadline - 10:
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    
    # Secondary optimizations from perturbations
    np.random.seed(42)
    for i in range(15):
        if time.time() > deadline - 5:
            break
        jitter = np.random.normal(0, 0.1, 3)
        c0 = seed + jitter
        res = minimize(obj, c0, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000, 'maxfev': 1000})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
            
    return best_c
