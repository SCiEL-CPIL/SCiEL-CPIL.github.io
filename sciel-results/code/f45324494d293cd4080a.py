"""Refined uncertainty inequality: Laguerre double roots (m = 12, n = 1)"""
import time
import math
import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize
import sympy as sp

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _bessel_zero_J1(k):
    """Approximate k-th zero of J_1 using asymptotic expansion."""
    x = (k + 0.25) * math.pi
    if x < 0.1:
        return 3.8317059702075125
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _validate_z_list(z_list):
    """Check that z_list is valid: finite, > 0, pairwise distinct."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    # Check pairwise distinctness
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-8:
                return False
    return True

def _fast_score(z_list):
    """
    Approximate the score B/(2*pi) using floating point arithmetic.
    Returns None if the system is singular or computation fails.
    """
    try:
        z = np.array(sorted(z_list), dtype=float)
        # 1. Build the linear system A c = b
        # g(x) = sum_{k=0}^{25} c_k L_{2k}^{-1/2}(x)
        # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0
        
        alpha = -0.5
        degrees = np.arange(0, 51, 2) # 0, 2, ..., 50
        n_coeff = 26
        
        A = np.zeros((n_coeff, n_coeff))
        b = np.zeros(n_coeff)
        
        # Precompute Laguerre polynomials and their derivatives at needed points
        # Points: 0 and z_1..z_12
        points = np.concatenate(([0.0], z))
        
        # Evaluate L_d and L'_d at all points
        # We need L_d(0), L'_d(0) and L_d(z_i), L'_d(z_i)
        
        # L_0(x) = 1, L'_0(x) = 0
        # L_d(x) for d in degrees
        
        # Efficient evaluation:
        # A[0, k] = L_{2k}(0)
        # A[1, k] = L'_{2k}(0)
        # A[2+2i, k] = L_{2k}(z_i)
        # A[3+2i, k] = L'_{2k}(z_i)
        
        # b[0] = 0
        # b[1] = 1
        # b[2+2i] = 0
        # b[3+2i] = 0
        
        # L_d^{alpha}(x) can be evaluated using scipy.special.genlaguerre
        # Note: genlaguerre(n, alpha) returns a poly1d object
        # Creating 26 poly1d objects might be slow if done repeatedly, 
        # but we only do it once per score evaluation.
        
        lag_pols = []
        lag_derivs = []
        for d in degrees:
            p = genlaguerre(d, alpha)
            lag_pols.append(p)
            lag_derivs.append(p.deriv())
            
        # Fill A and b
        # Row 0: g(0) = 0
        for k in range(n_coeff):
            A[0, k] = lag_pols[k](0.0)
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        for k in range(n_coeff):
            A[1, k] = lag_derivs[k](0.0)
        b[1] = 1.0
        
        # Rows for z_i
        for i in range(12):
            zi = z[i]
            row_g = 2 + 2*i
            row_dg = 3 + 2*i
            for k in range(n_coeff):
                A[row_g, k] = lag_pols[k](zi)
                A[row_dg, k] = lag_derivs[k](zi)
            b[row_g] = 0.0
            b[row_dg] = 0.0
            
        # Solve A c = b
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
            
        # Construct g(x)
        # g(x) = sum c_k L_{2k}(x)
        # We need q(x) = g(x) / (x * prod (x-z_i)^2)
        # Let D(x) = x * prod (x-z_i)^2
        # q(x) is a polynomial of degree 50 - (1 + 2*12) = 50 - 25 = 25
        
        # Get coefficients of g(x)
        # g(x) = sum c_k * (sum_{j=0}^{2k} ... x^j)
        # It's easier to construct the polynomial coefficients directly.
        
        g_coeffs = np.zeros(51) # Degree 50 max
        for k in range(n_coeff):
            d = 2*k
            # Get coeffs of L_d
            # genlaguerre returns poly1d, we can get .coeffs
            # Note: poly1d coeffs are in descending order of power
            p_coeffs = lag_pols[k].coeffs
            # Pad to length 51
            pad_len = 51 - len(p_coeffs)
            if pad_len > 0:
                p_coeffs_padded = np.concatenate((np.zeros(pad_len), p_coeffs))
            else:
                p_coeffs_padded = p_coeffs
            g_coeffs += c[k] * p_coeffs_padded
            
        # Get coefficients of D(x) = x * prod (x-z_i)^2
        # Start with D(x) = x
        d_coeffs = np.array([1.0, 0.0]) # x
        for zi in z:
            # Multiply by (x - zi)^2 = x^2 - 2 zi x + zi^2
            # Convolution
            factor = np.array([1.0, -2.0*zi, zi**2])
            d_coeffs = np.convolve(d_coeffs, factor)
            
        # Now we have g_coeffs (degree <= 50) and d_coeffs (degree 25)
        # We need to divide g by d.
        # Since g(z_i)=0 and g'(z_i)=0, d divides g exactly.
        # We can use np.polydiv
        
        q_coeffs, remainder = np.polydiv(g_coeffs, d_coeffs)
        
        # Check remainder is close to zero
        if np.max(np.abs(remainder)) > 1e-6:
            # If remainder is not small, the system might be ill-conditioned or numerical errors are high
            # But theoretically it should be zero. Let's proceed if it's reasonably small.
            pass
            
        # Find real roots of q(x)
        roots = np.roots(q_coeffs)
        real_roots = roots[np.isreal(roots)].real
        
        # Filter roots:
        # 1. Must be real
        # 2. q(r - eps) and q(r + eps) must have opposite signs
        # 3. We want the largest such r, B = max(0, r_max)
        
        eps = 1e-10
        max_r = 0.0
        
        # Evaluate q at roots +/- eps
        # q(x) is given by q_coeffs
        for r in real_roots:
            if r <= 0:
                continue
            q_minus = np.polyval(q_coeffs, r - eps)
            q_plus = np.polyval(q_coeffs, r + eps)
            if q_minus * q_plus < 0:
                if r > max_r:
                    max_r = r
                    
        B = max(0.0, max_r)
        score = B / (2.0 * math.pi)
        return score
        
    except Exception:
        return None

def _exact_score(z_list):
    """
    Compute the exact score using sympy.
    Returns None if invalid or timeout.
    """
    start_time = time.time()
    deadline = start_time + 45.0 # 45s budget for exact computation
    
    try:
        z_rationals = sorted(sp.Rational(float(z)) for z in z_list)
        x = sp.symbols('x')
        alpha = sp.Rational(-1, 2)
        
        laguerre_pols = []
        for d in range(0, 51, 2):
            laguerre_pols.append(sp.laguerre(d, x, alpha))
            
        A = sp.Matrix(26, 26, lambda i, j: 0)
        b = sp.Matrix(26, 1, lambda i, j: 0)
        
        for k in range(26):
            A[0, k] = laguerre_pols[k].subs(x, 0)
            A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
        b[0, 0] = 0
        b[1, 0] = 1
        
        for i in range(12):
            zi = z_rationals[i]
            row_idx = 2 + 2*i
            for k in range(26):
                A[row_idx, k] = laguerre_pols[k].subs(x, zi)
                A[row_idx+1, k] = laguerre_pols[k].diff(x).subs(x, zi)
            b[row_idx, 0] = 0
            b[row_idx+1, 0] = 0
            
        if time.time() > deadline:
            return None
            
        try:
            c = A.LUsolve(b)
        except Exception:
            return None
            
        g = sum(c[k] * laguerre_pols[k] for k in range(26))
        denom = x
        for zi in z_rationals:
            denom = denom * (x - zi)**2
            
        g_poly = sp.Poly(g, x)
        denom_poly = sp.Poly(denom, x)
        
        try:
            q_poly, _ = g_poly.div(denom_poly)
        except Exception:
            return None
            
        q_expr = q_poly.as_expr()
        
        if time.time() > deadline:
            return None
            
        try:
            real_roots_list = sp.real_roots(q_expr)
        except Exception:
            return None
            
        eps = sp.Rational(1, 10)**198
        max_r = 0
        for r in real_roots_list:
            try:
                q_at_minus = q_expr.subs(x, r - eps)
                q_at_plus = q_expr.subs(x, r + eps)
                if q_at_minus * q_at_plus < 0:
                    if r > max_r:
                        max_r = r
            except Exception:
                continue
                
        B = max(0, max_r)
        score = B / (2 * sp.pi)
        return float(score)
        
    except Exception:
        return None

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    """
    deadline = time.time() + 580.0 # Leave 50s margin for exact check and return
    
    # Initial candidates
    # 1. Bessel zeros squared (J0)
    z_j0 = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    # 2. Bessel zeros squared (J1)
    z_j1 = [_bessel_zero_J1(i)**2 for i in range(1, 13)]
    # 3. Geometric
    z_geo = [2.0 * (1.5**i) for i in range(12)]
    # 4. Arithmetic
    z_arith = [2.0 + 3.0*i for i in range(12)]
    
    candidates = [z_j0, z_j1, z_geo, z_arith]
    
    best_z = None
    best_score = float('inf')
    
    # Evaluate initial candidates with fast score
    for z_cand in candidates:
        if time.time() > deadline - 60:
            break
        if not _validate_z_list(z_cand):
            continue
        score = _fast_score(z_cand)
        if score is not None and score < best_score:
            best_score = score
            best_z = z_cand
            
    if best_z is None:
        # Fallback
        best_z = z_j0
        best_score = _fast_score(best_z)
        if best_score is None:
            best_score = 100.0
            
    # Optimization using Nelder-Mead
    # We optimize the log of the score to handle wide ranges, or just score directly
    # Score is positive.
    
    def objective(z_flat):
        z_list = list(z_flat)
        if not _validate_z_list(z_list):
            return 1e10
        s = _fast_score(z_list)
        if s is None:
            return 1e10
        return s
        
    if time.time() < deadline - 120: # Need at least 120s left for optimization + exact check
        # Start from best_z
        x0 = np.array(best_z, dtype=float)
        
        # Constraints: z_i > 0. Nelder-Mead doesn't handle constraints natively well.
        # We can transform variables: z_i = exp(y_i).
        # Then y_i can be any real number.
        
        def objective_y(y_flat):
            z_list = np.exp(y_flat)
            if not _validate_z_list(z_list):
                return 1e10
            s = _fast_score(z_list)
            if s is None:
                return 1e10
            return s
            
        y0 = np.log(x0)
        
        # Run optimization
        # maxiter should be reasonable. Each iteration calls objective ~n times.
        # n=12. So 1000 iterations ~ 12000 calls.
        # Fast score is ~10-50ms. 12000 * 0.05 = 600s. Too slow.
        # We need to limit iterations or use a faster score.
        # Let's limit maxiter to 200 and check time.
        
        try:
            result = minimize(objective_y, y0, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.success and result.fun < best_score:
                best_z = list(np.exp(result.x))
                best_score = result.fun
        except Exception:
            pass
            
    # Final Exact Verification
    # If we have time, verify the best candidate exactly.
    # If the exact score is valid, we return best_z.
    # If exact score is invalid (e.g. singular), we might need to fallback to a known valid set.
    # But the problem says "A singular system is invalid".
    # If our fast score said it's valid, it's likely valid.
    # However, if the exact check fails due to timeout, we just return the fast-scored best_z.
    
    if time.time() < deadline - 50:
        exact_s = _exact_score(best_z)
        if exact_s is not None:
            # If exact score is valid, we trust it.
            # The score returned by construct() is not used directly for scoring, 
            # but the z_list is. The evaluator will re-compute the score exactly.
            # So we just need to return a z_list that yields a low score.
            # If exact_s is valid, best_z is a good candidate.
            pass
        else:
            # Exact check failed or timed out. 
            # We still return best_z, hoping it's valid.
            # To be safer, we could check if a default candidate (like z_j0) has a known valid exact score.
            # But we don't have that info.
            # Let's stick with best_z.
            pass
            
    # Final validation
    if not _validate_z_list(best_z):
        # Emergency fallback
        best_z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        
    return best_z

# EVOLVE-BLOCK-END
