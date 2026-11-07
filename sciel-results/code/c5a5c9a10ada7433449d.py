import numpy as np
from scipy.optimize import minimize, differential_evolution, brentq
import time

def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as list [c0, c1, ..., cn]."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    H = [[1.0], [0.0, 2.0]]
    for k in range(2, n + 1):
        prev1 = H[k - 1]
        prev2 = H[k - 2]
        curr = [0.0] * (k + 1)
        for i in range(k):
            curr[i + 1] += 2.0 * prev1[i]
        for i in range(k - 1):
            curr[i] += -2.0 * (k - 1) * prev2[i]
        H.append(curr)
    return H[n]

# Precompute Hermite coefficients
_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)

# Convert to Q(y) coefficients
# g(x) = c0 H0 + c1 H4 + c2 H8 + c3 H12
# Q(y) = g(sqrt(y)) / y
# H0(x) = 1 -> 1. Not divisible by x^2.
# Wait, g(0)=0 implies c0*1 + c1*12 + c2*1344 + c3*H12(0) = 0.
# H0 has constant term. H4, H8, H12 have constant terms.
# g(x) = (c0*1) + (c1*12 + c2*1344 + c3*H12(0)) + (c1*(-48) + c2*(-13440) + c3*H12_2)x^2 + ...
# Since g(0)=0, the constant term is 0.
# So g(x) = a2 x^2 + a4 x^4 + ...
# Q(y) = a2 + a4 y + a6 y^2 + ...
# Coefficients of Q(y) are linear in c0, c1, c2.

# Let's extract the coefficients for x^2, x^4, ..., x^12 from H0, H4, H8, H12.
# H0: [1] -> x^0: 1
# H4: [12, -48, 16] (coeffs for x^0, x^2, x^4)
# H8: [1344, -13440, 11520, -3072, 256] (coeffs for x^0, x^2, x^4, x^6, x^8)
# H12: Need coeffs for x^0, x^2, ..., x^12.

def _get_even_coeffs(H):
    """Extract coefficients for x^0, x^2, ..., x^{2k} from H."""
    # H is [c0, c1, ..., cn]
    # Even terms are at indices 0, 2, 4, ...
    return [H[i] for i in range(0, len(H), 2)]

_H0_even = _get_even_coeffs(_H0) # [1]
_H4_even = _get_even_coeffs(_H4) # [12, -48, 16]
_H8_even = _get_even_coeffs(_H8) # [1344, -13440, 11520, -3072, 256]
_H12_even = _get_even_coeffs(_H12) # [H12(0), H12(2), ..., H12(12)]

# Q(y) = g(x)/x^2 where x^2=y.
# g(x) = sum_{k=0}^6 G_{2k} x^{2k}
# Q(y) = sum_{k=1}^6 G_{2k} y^{k-1}
# So Q(y) coeffs are [G_2, G_4, G_6, G_8, G_10, G_12]
# G_{2k} = c0 * (H0_{2k}) + c1 * (H4_{2k}) + c2 * (H8_{2k}) + c3 * (H12_{2k})
# Note: H0_{2k} is 0 for k>=1. H0_0 = 1.
# So G_0 = c0*1 + c1*H4_0 + c2*H8_0 + c3*H12_0 = 0 (by constraint)
# G_2 = c1*H4_2 + c2*H8_2 + c3*H12_2
# ...
# G_12 = c2*H8_12(0) + c3*H12_12. H8_12 is 0. So G_12 = c3*H12_12.

# Let's build the matrix M such that Q_coeffs = M @ [c0, c1, c2]
# c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0) / H12_0
# H0_0 = 1
# H4_0 = 12
# H8_0 = 1344
# H12_0 = _H12_even[0]

c3_c0 = -_H0_even[0] / _H12_even[0]
c3_c1 = -_H4_even[0] / _H12_even[0]
c3_c2 = -_H8_even[0] / _H12_even[0]

# Q(y) has degree 10 (y^10 corresponds to x^12)
# Q_coeffs[j] corresponds to y^j, which is x^{2j+2}
# So Q_coeffs[0] = G_2, Q_coeffs[1] = G_4, ..., Q_coeffs[10] = G_12

M = np.zeros((11, 3)) # 11 coeffs for Q(y), 3 variables c0, c1, c2
for j in range(11):
    # G_{2(j+1)} = c0*H0_{2j+2} + c1*H4_{2j+2} + c2*H8_{2j+2} + c3*H12_{2j+2}
    # H0_{2j+2} is 0 for j>=0 (since H0 is degree 0)
    h0_val = 0.0
    h4_val = _H4_even[j+1] if j+1 < len(_H4_even) else 0.0
    h8_val = _H8_even[j+1] if j+1 < len(_H8_even) else 0.0
    h12_val = _H12_even[j+1] if j+1 < len(_H12_even) else 0.0
    
    # Coeff for c0:
    M[j, 0] = h0_val + c3_c0 * h12_val
    # Coeff for c1:
    M[j, 1] = h4_val + c3_c1 * h12_val
    # Coeff for c2:
    M[j, 2] = h8_val + c3_c2 * h12_val

def _compute_value_fast(c0, c1, c2):
    """Compute value using linear map and robust root finding."""
    c = np.array([c0, c1, c2])
    q_coeffs = M @ c # [G_2, G_4, ..., G_12]
    
    # Define Q(y)
    # We want largest y > 0 such that Q(y) changes sign.
    # Q(y) is degree 10.
    
    # Find bounds for roots.
    # Use Cauchy bound or similar.
    # For now, just scan a reasonable range.
    # Roots of Q(y) are related to roots of g(x)/x^2.
    # g(x) is dominated by c3 H12(x) for large x.
    # H12(x) ~ 4096 x^12.
    # Q(y) ~ c3 * 4096 y^10.
    
    # Let's find sign changes by sampling.
    # Range of y: [0, Y_max]
    # If c3 is small, roots might be large.
    # Let's estimate Y_max.
    # If all roots are negative, value is inf? Or 0?
    # "largest real root ... at which q changes sign".
    # If no such root, return inf?
    
    # We want to MINIMIZE the value.
    # So we want small positive sign-changing roots.
    
    # Scan y from 0 to some max.
    # Let's use a log-spaced grid for wide range, then linear for local.
    
    # Determine Y_max dynamically.
    # If q_coeffs[10] is 0, degree is lower.
    
    # Simple robust method:
    # 1. Check if Q(0) = G_2 is 0? If so, y=0 is a root.
    #    But we need y>0.
    
    # Let's sample y in [1e-6, 1e6]
    # If a sign change is found, refine.
    
    # To be safe, let's check the leading coefficient.
    # If Q(y) > 0 for all y > 0, no roots.
    
    # We will use a grid search for sign changes.
    y_min = 1e-8
    y_max = 1e6
    
    # If the polynomial is always positive or always negative, no roots.
    # Let's evaluate at y=0 and y=y_max.
    
    def Q(y):
        # Horner's method
        res = 0.0
        for coeff in reversed(q_coeffs):
            res = res * y + coeff
        return res
    
    # Check Q(0)
    q0 = Q(0.0)
    
    # We need to find the LARGEST sign-changing root.
    # So we should scan from high y to low y?
    # Or find all and take max.
    
    # Let's find all sign-changing roots in [y_min, y_max].
    # Use a coarse grid first.
    ys = np.logspace(np.log10(y_min), np.log10(y_max), 1000)
    Qs = np.array([Q(y) for y in ys])
    
    # Find sign changes
    sign_changes = []
    for i in range(len(ys) - 1):
        if Qs[i] * Qs[i+1] < 0:
            # Root in [ys[i], ys[i+1]]
            try:
                root = brentq(Q, ys[i], ys[i+1], xtol=1e-12)
                if root > 0:
                    sign_changes.append(root)
            except:
                pass
    
    # Also check if Q(0) == 0?
    # If Q(0)==0, is it a sign change?
    # q(x) = Q(x^2). If Q(0)=0, q(x) has root at x=0.
    # Task: "largest real root of q".
    # Usually implies non-zero?
    # "r be the largest real root of q at which q changes sign".
    # If q(0)=0 and sign changes, r=0? Value 0?
    # But g(0)=0 is always true.
    # g(x) = x^2 Q(x^2).
    # q(x) = Q(x^2).
    # q(0) = Q(0).
    # If Q(0) != 0, q(0) != 0.
    # If Q(0) == 0, q(0) = 0.
    # Does q change sign at 0?
    # q(x) = a x^2 + ...
    # If a=0, q(x) = b x^4 + ...
    # If b>0, q(x) > 0 near 0. No sign change.
    # If b<0, q(x) < 0 near 0. No sign change.
    # So root at 0 is not a sign-changing root unless higher order terms dominate?
    # Actually, if q(x) ~ x^k, it doesn't change sign at 0 if k is even.
    # Q(y) ~ y^m. q(x) ~ x^{2m}. Even power. No sign change at 0.
    # So we only care about y > 0.
    
    if not sign_changes:
        # Maybe the root is outside [y_min, y_max]?
        # If Q(y_max) and Q(0) have different signs, there is a root.
        # If Q(y_max) and Q(0) have same sign, there could be 2, 4... roots.
        # If we missed them, we return inf?
        # To be safe, if no roots found, return inf.
        return float('inf')
    
    largest_root = max(sign_changes)
    return largest_root / (2 * np.pi)

def construct():
    start_time = time.time()
    deadline = 200.0

    seed_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240.0])
    best_c = seed_c.copy()
    best_val = _compute_value_fast(best_c[0], best_c[1], best_c[2])
    
    if not np.isfinite(best_val):
        # If seed is invalid in our fast calc, try to fix or find valid one
        # But seed should be valid.
        pass

    def elapsed():
        return time.time() - start_time

    # Phase 1: Differential Evolution
    if elapsed() < deadline - 70:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            de_result = differential_evolution(
                lambda x: _compute_value_fast(x[0], x[1], x[2]), bounds,
                maxiter=100, popsize=30, tol=1e-8, seed=42, polish=False
            )
            v = _compute_value_fast(de_result.x[0], de_result.x[1], de_result.x[2])
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_c = de_result.x.copy()
        except:
            pass

    # Phase 2: Local Optimization
    np.random.seed(123)
    for i in range(50):
        if elapsed() > deadline - 20:
            break
        
        if i == 0:
            x0 = best_c.copy()
        else:
            x0 = best_c + np.random.normal(0, 0.5, size=3)
        
        try:
            result = minimize(
                lambda x: _compute_value_fast(x[0], x[1], x[2]), x0, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            v = _compute_value_fast(result.x[0], result.x[1], result.x[2])
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_c = result.x.copy()
        except:
            pass

    # Final check
    final_val = _compute_value_fast(best_c[0], best_c[1], best_c[2])
    if np.isfinite(final_val) and final_val < best_val:
        best_val = final_val

    return best_c
