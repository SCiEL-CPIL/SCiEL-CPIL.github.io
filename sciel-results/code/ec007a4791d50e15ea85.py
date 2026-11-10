"""Bootstrap spawn: FAST SEARCH approach – use numerical approximation to explore search space rapidly."""
import time
import math
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import binom as sp_binom

# EVOLVE-BLOCK-START

# Precompute Laguerre coefficients C[j, k] for L_{2j}^{(-1/2)}(x)
# j=0..25, n=2j
_C_LAGUERR = np.zeros((26, 51))
_alpha = -0.5
for _j in range(26):
    _n = 2 * _j
    for _k in range(_n + 1):
        _C_LAGUERR[_j, _k] = sp_binom(_n + _alpha, _n - _k) * ((-1.0)**_k) / math.factorial(_k)

def _compute_score_fast(z_list):
    """
    Fast numerical approximation of the score B/(2*pi).
    z_list: list of 12 floats, assumed sorted and distinct.
    Returns score or a large penalty if invalid.
    """
    try:
        z = np.array(z_list, dtype=float)
        if np.any(z <= 0) or np.any(np.diff(z) < 1e-8):
            return 1e100
        
        C = _C_LAGUERR
        
        # Build system A c = b
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        # L_n(0) = binom(n+alpha, n)
        for j in range(26):
            n = 2 * j
            A[0, j] = sp_binom(n + _alpha, n)
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        # L_n'(0) = coeff of x^1 in L_n = binom(n+alpha, n-1) * (-1)^1 / 1!
        for j in range(26):
            n = 2 * j
            if n >= 1:
                A[1, j] = -sp_binom(n + _alpha, n - 1)
            else:
                A[1, j] = 0.0
        b[1] = 1.0
        
        # Rows 2..25: g(z_i)=0, g'(z_i)=0
        for i in range(12):
            zi = z[i]
            row_g = 2 + 2 * i
            row_dp = 3 + 2 * i
            
            # Precompute powers of zi
            powers = np.zeros(51)
            powers[0] = 1.0
            for k in range(1, 51):
                powers[k] = powers[k-1] * zi
            
            for j in range(26):
                n = 2 * j
                # g(zi) = sum c_j L_{2j}(zi) = sum c_j sum_k C[j,k] zi^k
                A[row_g, j] = np.dot(C[j, :n+1], powers[:n+1])
                
                # g'(zi) = sum c_j L'_{2j}(zi)
                # L'_{2j}(zi) = sum_{k=1}^n k C[j,k] zi^{k-1}
                # Let m = k-1. sum_{m=0}^{n-1} (m+1) C[j, m+1] zi^m
                if n > 0:
                    coeffs_deriv = np.arange(1, n+1) * C[j, 1:n+1]
                    A[row_dp, j] = np.dot(coeffs_deriv, powers[:n])
                else:
                    A[row_dp, j] = 0.0
            
            b[row_g] = 0.0
            b[row_dp] = 0.0
        
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e100
            
        # Construct g(x) coefficients
        # g(x) = sum_j c_j L_{2j}(x) = sum_j c_j sum_k C[j,k] x^k
        # g_coeffs[k] = sum_j c_j C[j,k]
        g_coeffs = np.dot(c, C) # shape (51,)
        
        # Construct D(x) = x * prod_i (x - z_i)^2
        # P(x) = prod_i (x - z_i)
        p_coeffs = np.array([1.0])
        for zi in z:
            # multiply by (x - zi)
            new_p = np.zeros(len(p_coeffs) + 1)
            new_p[1:] += p_coeffs
            new_p[:-1] -= p_coeffs * zi
            p_coeffs = new_p
        
        # P(x)^2
        p2_coeffs = np.convolve(p_coeffs, p_coeffs) # degree 24
        
        # D(x) = x * P(x)^2 => shift by 1
        d_coeffs = np.zeros(26) # degree 25
        d_coeffs[1:] = p2_coeffs
        
        # Polynomial division g / D
        # g has degree 50, D has degree 25. q has degree 25.
        # We want q such that g = q * D + R
        # Since g is constructed to have double roots at z_i, R should be 0.
        # But numerically, it might not be exactly 0. We just take the quotient.
        
        g_work = g_coeffs.copy()
        q_coeffs = np.zeros(26)
        
        # Long division - track current degree
        # Leading coeff of D is 1 (from x * x^24).
        for k in range(25, -1, -1):
            current_deg = 50 - (25 - k)  # = 25 + k
            if abs(g_work[current_deg]) < 1e-15:
                continue
            factor = g_work[current_deg] / d_coeffs[25]
            q_coeffs[k] = factor
            # Subtract factor * D(x) * x^(current_deg - 25)
            shift = current_deg - 25  # = k
            g_work[shift:shift+26] -= factor * d_coeffs
        
        # Find real roots of q
        try:
            roots = np.roots(q_coeffs)
        except Exception:
            return 1e100
            
        # Filter real roots
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        if len(real_roots) == 0:
            return 0.0
            
        # Check sign change for each real root
        eps = 1e-12
        max_r = 0
        for r in real_roots:
            q_minus = np.polyval(q_coeffs, r - eps)
            q_plus = np.polyval(q_coeffs, r + eps)
            if q_minus * q_plus < 0:
                if r > max_r:
                    max_r = r
                    
        B = max(0, max_r)
        score = B / (2 * math.pi)
        return score
        
    except Exception:
        return 1e100

def _unconstrained_to_z(u):
    """Map unconstrained positive u to sorted z."""
    z = np.cumsum(u)
    return z

def _objective(u):
    """Objective function for optimization. u are positive increments."""
    z = _unconstrained_to_z(u)
    return _compute_score_fast(z)

def construct():
    """Return 12 double-root locations using FAST SEARCH."""
    start_time = time.time()
    deadline = start_time + 580.0  # 630s kill, keep 50s margin

    # Initial best: Bessel zeros squared
    def _bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

    z_baseline = np.array([_bessel_zero_J0(i+1)**2 for i in range(12)])
    best_z = z_baseline.copy()
    best_score = _compute_score_fast(best_z)

    # Differential Evolution
    # Bounds for u_i: [1e-3, 100]
    bounds = [(1e-3, 100.0)] * 12
    
    # Run DE with time-limited iterations
    de_maxiter = 50
    if time.time() < deadline - 200:
        de_maxiter = 80
    if time.time() < deadline - 300:
        de_maxiter = 120
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=de_maxiter, 
            popsize=12,
            tol=1e-5,
            seed=42,
            polish=False,
        )
        
        if res.success or res.fun < best_score:
            u_best = res.x
            z_candidate = _unconstrained_to_z(u_best)
            score_candidate = _compute_score_fast(z_candidate)
            if score_candidate < best_score:
                best_z = z_candidate
                best_score = score_candidate
    except Exception:
        pass

    # Local Polish from multiple starting points
    start_points = [np.diff(np.concatenate(([0], best_z)))]
    
    # Add some random perturbations of the baseline
    for _ in range(8):
        if time.time() > deadline - 150:
            break
        z_pert = z_baseline * (1 + 0.15 * np.random.randn(12))
        z_pert = np.sort(z_pert)
        if np.all(z_pert > 0) and np.all(np.diff(z_pert) > 1e-6):
            u_pert = np.diff(np.concatenate(([0], z_pert)))
            u_pert = np.clip(u_pert, 1e-3, 100.0)
            start_points.append(u_pert)
            
    # Add DE result if available
    if 'res' in locals() and res.x is not None:
        u_de = np.diff(np.concatenate(([0], _unconstrained_to_z(res.x))))
        u_de = np.clip(u_de, 1e-3, 100.0)
        start_points.append(u_de)

    for u_start in start_points:
        if time.time() > deadline - 80:
            break
        
        # Nelder-Mead
        try:
            res_nm = minimize(
                _objective, 
                u_start, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if res_nm.fun < best_score:
                u_polished = res_nm.x
                z_polished = _unconstrained_to_z(u_polished)
                score_polished = _compute_score_fast(z_polished)
                if score_polished < best_score:
                    best_z = z_polished
                    best_score = score_polished
        except Exception:
            pass
            
        # Powell (often better for smooth-ish landscapes)
        if time.time() > deadline - 80:
            break
        try:
            res_pw = minimize(
                _objective, 
                u_start, 
                method='Powell', 
                options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            
            if res_pw.fun < best_score:
                u_polished = res_pw.x
                z_polished = _unconstrained_to_z(u_polished)
                score_polished = _compute_score_fast(z_polished)
                if score_polished < best_score:
                    best_z = z_polished
                    best_score = score_polished
        except Exception:
            pass

    # Final check: ensure z is valid
    if np.any(best_z <= 0) or np.any(np.diff(best_z) < 1e-8):
        return z_baseline.tolist()
        
    return best_z.tolist()

# EVOLVE-BLOCK-END
