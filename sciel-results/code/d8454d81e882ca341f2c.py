"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from sympy import Rational, Matrix, Poly, real_roots, pi as sympy_pi, binomial, symbols, factor, laguerre as _sympy_laguerre

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728  # j_{0,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _compute_laguerre_values(points, max_n, alpha):
    """
    Compute L_n^(alpha)(p) for n=0..max_n for each p in points.
    Returns a list of lists: vals[n][i] = L_n^(alpha)(points[i]).
    Uses recurrence for speed.
    """
    num_points = len(points)
    vals = [[Rational(0)] * num_points for _ in range(max_n + 1)]
    
    # L_0^(alpha)(x) = 1
    for i in range(num_points):
        vals[0][i] = Rational(1)
        
    if max_n >= 1:
        # L_1^(alpha)(x) = -x + alpha + 1
        for i in range(num_points):
            vals[1][i] = -points[i] + alpha + 1
            
    for n in range(1, max_n):
        # (n+1) L_{n+1} = (2n + alpha + 1 - x) L_n - (n + alpha) L_{n-1}
        for i in range(num_points):
            x = points[i]
            term1 = (2*n + alpha + 1 - x) * vals[n][i]
            term2 = (n + alpha) * vals[n-1][i]
            vals[n+1][i] = (term1 - term2) / (n + 1)
            
    return vals

def _compute_score_exact(z_list, deadline=None):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    if deadline is not None and time.time() > deadline:
        return None
        
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Points to evaluate: 0, z_1, ..., z_12
    points = [Rational(0)] + z_rationals
    
    # Pre-compute Laguerre values and derivatives
    # We need L_{2k}(x) and L'_{2k}(x) for k=0..25 (n=0..50)
    # L'_{n}(x) = -L_{n-1}^{(alpha+1)}(x)
    
    # Compute L_n^(alpha) for n=0..50
    max_n = 50
    vals_alpha = _compute_laguerre_values(points, max_n, alpha)
    
    # Compute L_n^(alpha+1) for n=0..49 (needed for derivatives of L_1..L_50)
    # L'_n^(alpha)(x) = -L_{n-1}^{(alpha+1)}(x)
    # So for n=1..50, we need L_{n-1}^{(alpha+1)}
    # This means we need L_0^(alpha+1) .. L_49^(alpha+1)
    vals_alpha_plus_1 = _compute_laguerre_values(points, max_n - 1, alpha + 1)
    
    # Build matrix A and vector b
    # Variables c_0 ... c_25
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Row 0: g(0) = 0
    # g(0) = sum c_k L_{2k}(0)
    for k in range(26):
        n = 2*k
        A[0, k] = vals_alpha[n][0]
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    # g'(0) = sum c_k L'_{2k}(0) = sum c_k (-L_{2k-1}^{(alpha+1)}(0))
    # For k=0, L'_0 = 0. For k>=1, L'_{2k} = -L_{2k-1}^{(alpha+1)}
    for k in range(26):
        n = 2*k
        if n == 0:
            A[1, k] = 0
        else:
            # Index in vals_alpha_plus_1 is n-1
            A[1, k] = -vals_alpha_plus_1[n-1][0]
    b[1, 0] = 1
    
    # Rows 2..25: for each z_i (index i-1 in z_rationals, index i in points)
    for i in range(12):
        point_idx = i + 1
        # g(z_i) = 0
        row_idx = 2 + 2*i
        for k in range(26):
            n = 2*k
            A[row_idx, k] = vals_alpha[n][point_idx]
        b[row_idx, 0] = 0
        
        # g'(z_i) = 0
        row_idx = 3 + 2*i
        for k in range(26):
            n = 2*k
            if n == 0:
                A[row_idx, k] = 0
            else:
                A[row_idx, k] = -vals_alpha_plus_1[n-1][point_idx]
        b[row_idx, 0] = 0
        
    # Solve
    try:
        c = A.LUsolve(b)
    except Exception:
        return None  # singular system
    
    # Construct g(x) coefficients in monomial basis
    # We need the monomial coefficients of L_n^(alpha)(x).
    # It is faster to just build the polynomial g(x) directly by summing c_k * L_{2k}(x)
    # But we need the monomial coefficients for division.
    # Let's compute the monomial coefficients of L_n^(alpha)(x) once.
    
    # L_n^(alpha)(x) = sum_{j=0}^n binomial(n+alpha, n-j) (-x)^j / j!
    # We can compute these coefficients for n=0..50.
    
    lag_coeffs = []
    for n in range(51):
        coeffs = [Rational(0)] * (n + 1)
        for j in range(n + 1):
            coeffs[j] = binomial(n + alpha, n - j) * ((-1) ** j) / math.factorial(j)
        lag_coeffs.append(coeffs)
        
    max_deg = 50
    G = [Rational(0)] * (max_deg + 1)
    for k in range(26):
        ck = c[k]
        n = 2*k
        coeffs_k = lag_coeffs[n]
        for j in range(len(coeffs_k)):
            G[j] += ck * coeffs_k[j]
            
    # Construct denominator D(x) = x * prod (x - z_i)^2
    D = [Rational(0)] * (2 * 12 + 1)
    D[1] = 1 # x
    for zi in z_rationals:
        new_D = [Rational(0)] * (len(D) + 2)
        for i in range(len(D)):
            new_D[i] += D[i] * (zi**2)
            new_D[i+1] += D[i] * (-2 * zi)
            new_D[i+2] += D[i] * 1
        D = new_D
        
    # Polynomial division G / D using sympy
    G_poly = Poly(sum(G[i] * x**i for i in range(len(G))), x)
    D_poly = Poly(sum(D[i] * x**i for i in range(len(D))), x)
    
    try:
        q_poly, rem = G_poly.div(D_poly)
    except Exception:
        return None
        
    # Find real roots of q
    try:
        real_roots_list = real_roots(q_poly)
    except Exception:
        return None
        
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        try:
            q_at_minus = q_poly.eval(r - eps)
            q_at_plus = q_poly.eval(r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def _time_remaining(deadline):
    return deadline - time.time()

def _is_valid_candidate(z_list):
    """Check if candidate z_list is valid (positive, finite, distinct)."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-10:
                return False
    return True

def construct():
    """Return 12 double-root locations z_1..z_12 using Bessel-zero-inspired parameterization."""
    deadline = time.time() + 570.0  # 630s kill, keep 60s margin for safety
    
    # Compute Bessel zeros of J_0 for k=1..12
    # These are natural candidates for roots of orthogonal polynomials
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    result = list(z_bessel)
    if not _is_valid_candidate(result):
        result = [4.0 * (i + 1) for i in range(12)]
    
    best_score = float('inf')
    
    # Initial evaluation
    if _time_remaining(deadline) > 120:
        score = _compute_score_exact(result, deadline)
        if score is not None:
            best_score = score
    
    # Refinement strategy:
    # 1. Try global scaling factors
    # 2. Try local perturbations
    
    # Step 1: Global scaling
    # Bessel zeros are roughly proportional to (k-1/4)^2 * pi^2.
    # Scaling them might shift the "gap" where the root of q lies.
    scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 0.5]
    
    best_candidate = list(result)
    
    for s in scales:
        if _time_remaining(deadline) < 120:
            break
        candidate = [z * s for z in result]
        if not _is_valid_candidate(candidate):
            continue
        score = _compute_score_exact(candidate, deadline)
        if score is not None and score < best_score:
            best_score = score
            best_candidate = candidate
    
    result = best_candidate
    
    # Step 2: Local perturbations of the best candidate
    # We will try to adjust individual z_i to lower the score further.
    # Since the score is determined by the largest root of q, and q depends on all z_i,
    # we can try small random or structured perturbations.
    
    # Structured perturbations: shift each z_i by small amounts
    # We'll use a few iterations of this.
    
    max_iter = 20
    step_size = 0.05
    
    for _ in range(max_iter):
        if _time_remaining(deadline) < 120:
            break
        
        improved = False
        # Try perturbing each z_i
        for i in range(12):
            if _time_remaining(deadline) < 120:
                break
            
            # Try a few perturbations
            for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                if _time_remaining(deadline) < 120:
                    break
                
                candidate = list(result)
                # Absolute perturbation scaled by current value
                candidate[i] = result[i] * (1 + delta)
                
                if not _is_valid_candidate(candidate):
                    continue
                
                score = _compute_score_exact(candidate, deadline)
                if score is not None and score < best_score:
                    best_score = score
                    result = candidate
                    improved = True
                    # Reduce step size if we improved
                    step_size = max(0.01, step_size * 0.5)
                    break # Move to next i
                
        if not improved:
            # If no improvement with current step size, try smaller step or break
            step_size *= 0.5
            if step_size < 0.001:
                break

    return result

# EVOLVE-BLOCK-END
