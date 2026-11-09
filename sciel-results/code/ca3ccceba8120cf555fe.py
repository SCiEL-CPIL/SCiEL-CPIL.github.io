"""Refined program for Laguerre double roots (m=12, n=1)."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _laguerre_coeffs(n, alpha):
    """
    Return the coefficients of the generalized Laguerre polynomial L_n^(alpha)(x)
    as a numpy array [c_n, c_{n-1}, ..., c_0].
    """
    if n == 0:
        return np.array([1.0])
    
    # Use recurrence: L_{k+1} = ((2k+alpha+1-x)L_k - (k+alpha)L_{k-1})/(k+1)
    L_prev = np.array([1.0])
    L_curr = np.array([1.0 + alpha, -1.0])  # L_1(x) = -x + (1+alpha)
    
    if n == 1:
        return L_curr
    
    for k in range(1, n):
        term1 = np.convolve(L_curr, [-1.0, 2.0 * k + alpha + 1.0])
        term2 = (k + alpha) * L_prev
        
        max_len = max(len(term1), len(term2))
        term1_padded = np.zeros(max_len)
        term2_padded = np.zeros(max_len)
        term1_padded[:len(term1)] = term1
        term2_padded[:len(term2)] = term2
        
        L_next = (term1_padded - term2_padded) / (k + 1.0)
        L_prev = L_curr
        L_curr = L_next
    
    return L_curr

def _compute_score_float(z_list):
    """
    Fast floating point approximation of the score.
    """
    z = np.array(z_list)
    if len(z) != 12:
        return 1e10
    if np.any(z <= 0):
        return 1e10
    if np.any(np.diff(np.sort(z)) < 1e-12):
        return 1e10
    
    alpha = -0.5
    points = np.concatenate(([0.0], z))
    N_pts = len(points)
    
    L_minus = np.zeros((51, N_pts))
    L_plus = np.zeros((51, N_pts))
    
    L_minus[0, :] = 1.0
    L_plus[0, :] = 1.0
    L_minus[1, :] = 1.0 + (-0.5) - points
    L_plus[1, :] = 1.0 + 0.5 - points
    
    for k in range(1, 50):
        L_minus[k+1, :] = ((2.0 * k - 0.5 + 1.0 - points) * L_minus[k, :] - (k - 0.5) * L_minus[k-1, :]) / (k + 1.0)
        L_plus[k+1, :] = ((2.0 * k + 0.5 + 1.0 - points) * L_plus[k, :] - (k + 0.5) * L_plus[k-1, :]) / (k + 1.0)
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    for k in range(26):
        n = 2 * k
        A[0, k] = L_minus[n, 0]
        if n == 0:
            A[1, k] = 0.0
        else:
            A[1, k] = -L_plus[n-1, 0]
            
        for i in range(12):
            pt_idx = 1 + i
            A[2 + 2*i, k] = L_minus[n, pt_idx]
            if n == 0:
                A[3 + 2*i, k] = 0.0
            else:
                A[3 + 2*i, k] = -L_plus[n-1, pt_idx]
                
    b[0] = 0.0
    b[1] = 1.0
    
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e10
        
    g_coeffs = np.zeros(51)
    for k in range(26):
        n = 2 * k
        L_coeffs = _laguerre_coeffs(n, -0.5)
        g_coeffs[:len(L_coeffs)] += c[k] * L_coeffs
        
    D_coeffs = np.array([1.0, 0.0])
    for i in range(12):
        quad = np.array([1.0, -2.0 * z[i], z[i]**2])
        D_coeffs = np.convolve(D_coeffs, quad)
        
    try:
        q_coeffs, remainder = np.polydiv(g_coeffs, D_coeffs)
    except Exception:
        return 1e10
        
    roots = np.roots(q_coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
            
    if not real_roots:
        return 1e10
        
    B = max(real_roots)
    if B < 0:
        B = 0.0
        
    score = B / (2.0 * math.pi)
    return score

def _compute_score_exact(z_list, deadline):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import (symbols, Rational, Poly, diff, real_roots, pi as sympy_pi,
                       Matrix, laguerre as sympy_laguerre)
    
    if time.time() > deadline:
        return None
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    laguerre_pols = []
    for d in range(0, 51, 2):
        try:
            lp = sympy_laguerre(d, alpha, x)
            laguerre_pols.append(lp)
        except Exception:
            return None
    
    if time.time() > deadline:
        return None
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[0, 0] = 0
    b[1, 0] = 1
    
    for i in range(12):
        zi = z_rationals[i]
        row_val = 2 + 2 * i
        row_der = 3 + 2 * i
        for k in range(26):
            A[row_val, k] = laguerre_pols[k].subs(x, zi)
            A[row_der, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_val, 0] = 0
        b[row_der, 0] = 0
    
    if time.time() > deadline:
        return None
    
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, remainder = g_poly.div(denom_poly)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        if time.time() > deadline:
            return None
        try:
            q_at_minus = q_expr.subs(x, r - eps)
            q_at_plus = q_expr.subs(x, r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def _is_valid_z(z_list):
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-12:
                return False
    return True

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def construct():
    deadline = time.time() + 560.0
    
    # Initial guess: Bessel zeros squared
    z_bessel = [_bessel_zero_J0(k)**2 for k in range(1, 13)]
    
    if not _is_valid_z(z_bessel):
        z_bessel = [0.5 * (1.5 ** i) for i in range(12)]
        
    best_z = list(z_bessel)
    best_score_float = _compute_score_float(best_z)
    
    # Multiple optimization passes with different strategies
    def objective(u):
        z = np.exp(u)
        if not _is_valid_z(z):
            return 1e10
        return _compute_score_float(z)
        
    u0 = np.log(np.array(z_bessel))
    
    # Pass 1: Standard Nelder-Mead
    if time.time() < deadline - 120:
        try:
            res = minimize(objective, u0, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-12})
            if res.success:
                u_opt = res.x
                z_opt = np.exp(u_opt)
                if _is_valid_z(z_opt):
                    score_opt = _compute_score_float(z_opt)
                    if score_opt < best_score_float:
                        best_z = list(z_opt)
                        best_score_float = score_opt
        except Exception:
            pass
        
    # Pass 2: Coarse-to-fine local search around best
    if time.time() < deadline - 90:
        for scale in [0.1, 0.05, 0.02, 0.01, 0.005]:
            if time.time() > deadline - 60:
                break
            for i in range(12):
                if time.time() > deadline - 60:
                    break
                for sign in [1, -1]:
                    if time.time() > deadline - 60:
                        break
                    cand = list(best_z)
                    cand[i] = best_z[i] * (1 + sign * scale)
                    if not _is_valid_z(cand):
                        continue
                    s = _compute_score_float(cand)
                    if s < best_score_float:
                        best_z = cand
                        best_score_float = s
                        
    # Pass 3: Pairwise perturbations
    if time.time() < deadline - 60:
        for i in range(12):
            if time.time() > deadline - 45:
                break
            for j in range(i + 1, 12):
                if time.time() > deadline - 45:
                    break
                for di in [0.01, -0.01]:
                    for dj in [0.01, -0.01]:
                        if time.time() > deadline - 45:
                            break
                        cand = list(best_z)
                        cand[i] = best_z[i] * (1 + di)
                        cand[j] = best_z[j] * (1 + dj)
                        if not _is_valid_z(cand):
                            continue
                        s = _compute_score_float(cand)
                        if s < best_score_float:
                            best_z = cand
                            best_score_float = s
    
    # Final polish with very small steps
    if time.time() < deadline - 30:
        for _ in range(3):
            improved = False
            for i in range(12):
                if time.time() > deadline - 20:
                    break
                for scale in [0.001, 0.0005]:
                    for sign in [1, -1]:
                        if time.time() > deadline - 20:
                            break
                        cand = list(best_z)
                        cand[i] = best_z[i] * (1 + sign * scale)
                        if not _is_valid_z(cand):
                            continue
                        s = _compute_score_float(cand)
                        if s < best_score_float - 1e-15:
                            best_z = cand
                            best_score_float = s
                            improved = True
            if not improved:
                break
    
    # Exact evaluation of the best candidate
    final_score = None
    if time.time() < deadline - 60:
        final_score = _compute_score_exact(best_z, deadline - 30)
        
    # If time allows, try a few exact perturbations
    if time.time() < deadline - 90 and final_score is not None:
        for i in range(12):
            if time.time() > deadline - 60:
                break
            for step in [0.001, 0.0005]:
                for sign in [1, -1]:
                    if time.time() > deadline - 30:
                        break
                    cand = list(best_z)
                    cand[i] = best_z[i] * (1 + sign * step)
                    if not _is_valid_z(cand):
                        continue
                    try:
                        s = _compute_score_exact(cand, deadline - 10)
                        if s is not None and s < final_score:
                            final_score = s
                            best_z = cand
                    except Exception:
                        continue
                        
    return best_z

# EVOLVE-BLOCK-END
