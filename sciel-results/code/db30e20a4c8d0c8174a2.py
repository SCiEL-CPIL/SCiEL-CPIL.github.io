import time
import numpy as np
import sympy as sp

def _evaluate(z_list):
    """Full exact evaluation returning B value."""
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    
    # Build generalized Laguerre polynomials L_0, L_2, ..., L_50
    # g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
    c = sp.symbols('c0:26')
    
    laguerre_terms = []
    for k in range(26):
        d = 2 * k
        Ld = sp.genlaguerre(d, alpha, x)
        laguerre_terms.append(Ld)
    
    g = sum(c[k] * laguerre_terms[k] for k in range(26))
    gp = sp.diff(g, x)
    
    # Convert z_i to rationals
    z_rats = [sp.Rational(float(zi)) for zi in z_list]
    
    # Build system of 26 equations
    equations = []
    # g(0) = 0
    equations.append(g.subs(x, 0))
    # g'(0) = 1
    equations.append(gp.subs(x, 0) - 1)
    # For each z_i: g(z_i) = 0, g'(z_i) = 0
    for zi in z_rats:
        equations.append(g.subs(x, zi))
        equations.append(gp.subs(x, zi))
    
    # Solve linear system
    A = sp.Matrix([eq.as_poly(x).coeffs() for eq in equations])  # This won't work directly
    # Better: build coefficient matrix properly
    # Each equation is linear in c_0..c_25
    # Extract coefficients of each c_k in each equation
    
    A = sp.zeros(26, 26)
    b = sp.zeros(26, 1)
    
    for i, eq in enumerate(equations):
        for j in range(26):
            A[i, j] = eq.coeff(c[j])
        b[i] = eq.subs({c[j]: 0 for j in range(26)})
    
    # Check for singularity
    det_A = A.det()
    if det_A == 0:
        return None  # singular
    
    try:
        sol = A.LUsolve(b)
    except Exception:
        return None
    
    c_vals = [sol[i] for i in range(26)]
    
    # Build g with solved coefficients
    g_sol = sum(c_vals[k] * laguerre_terms[k] for k in range(26))
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rats:
        denom = denom * (x - zi)**2
    
    # Polynomial division: q = g / denom
    g_poly = sp.Poly(g_sol, x)
    denom_poly = sp.Poly(denom, x)
    
    try:
        q_poly, rem = sp.div(g_poly, denom_poly, domain=sp.QQ)
    except Exception:
        return None
    
    if rem != 0:
        # Not exact division - this shouldn't happen if roots are correct
        # But due to rational conversion, there might be issues
        return None
    
    q_expr = q_poly.as_expr()
    
    # Find real roots of q
    try:
        real_roots = sp.real_roots(q_poly, auto=True)
    except Exception:
        return None
    
    B = 0
    for r in real_roots:
        r_val = float(r)
        if not np.isfinite(r_val) or r_val <= 0:
            continue
        # Check sign change at r - 1e-198 and r + 1e-198
        eps = sp.Rational(1, 10)**198
        r_minus = r - eps
        r_plus = r + eps
        
        q_minus = q_expr.subs(x, r_minus)
        q_plus = q_expr.subs(x, r_plus)
        
        q_minus_val = float(q_minus)
        q_plus_val = float(q_plus)
        
        if np.isfinite(q_minus_val) and np.isfinite(q_plus_val):
            if q_minus_val * q_plus_val < 0:
                B = max(B, r_val)
    
    return B

def construct():
    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin
    
    # Initial guess: spread out positive values
    # Try a few different strategies within the time budget
    
    best_B = float('inf')
    best_z = None
    
    # Strategy: try different configurations
    # Configuration 1: equally spaced starting from small values
    configs = []
    
    # Small clustered values
    for base in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for spread in [0.1, 0.5, 1.0, 2.0]:
            z = [base + i * spread for i in range(12)]
            if all(zi > 0 for zi in z) and len(set(z)) == 12:
                configs.append(z)
    
    # Geometric spacing
    for ratio in [1.1, 1.2, 1.5, 2.0]:
        for start in [0.1, 0.5, 1.0]:
            z = [start * (ratio ** i) for i in range(12)]
            if all(zi > 0 for zi in z) and len(set(z)) == 12:
                configs.append(z)
    
    # Random-ish but deterministic
    import random
    random.seed(42)
    for _ in range(20):
        z = sorted([random.uniform(0.01, 20.0) for _ in range(12)])
        # Ensure distinct
        if len(set(z)) == 12 and all(zi > 0 for zi in z):
            configs.append(z)
    
    # Sort configs by some heuristic - maybe prefer smaller values first
    # Since B is the max root of q, smaller z_i might lead to smaller B
    
    for z in configs:
        if time.time() > deadline - 30:  # Need at least 30s for evaluation
            break
        
        try:
            B = _evaluate(z)
            if B is not None and B < best_B:
                best_B = B
                best_z = z
        except Exception:
            continue
    
    if best_z is not None:
        # Ensure output format
        return [float(zi) for zi in best_z]
    else:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]
