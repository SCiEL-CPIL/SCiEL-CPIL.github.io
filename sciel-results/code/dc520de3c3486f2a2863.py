import time
import math
import sympy as sp
from sympy.polys.orthopoly import genlaguerre

# EVOLVE-BLOCK-START

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, S
    
    # Convert to rationals and sort
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = S(-1)/2
    
    # Build L_{2k}^{alpha}(x) for k=0..25
    laguerre_pols = []
    for k in range(26):
        n = 2*k
        laguerre_pols.append(genlaguerre(n, alpha)(x))
        
    # Build system matrix A (26x26) and vector b (26x1)
    A = Matrix(26, 26, lambda i, j: S(0))
    b = Matrix(26, 1, lambda i, j: S(0))
    
    # Precompute values for speed
    for k in range(26):
        L = laguerre_pols[k]
        Ld = L.diff(x)
        A[0, k] = L.subs(x, 0)
        A[1, k] = Ld.subs(x, 0)
        for i in range(12):
            zi = z_rationals[i]
            A[2 + 2*i, k] = L.subs(x, zi)
            A[3 + 2*i, k] = Ld.subs(x, zi)
            
    b[1, 0] = 1 # g'(0) = 1
    
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
        
    # Build g(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    # Denominator: x * prod (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
        
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    try:
        q_poly, rem = g_poly.div(denom_poly)
        if rem != 0:
            return None
    except Exception:
        return None
        
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
        
    eps = Rational(1, 10)**198
    max_r = S(0)
    for r in real_roots_list:
        try:
            qm = q_expr.subs(x, r - eps)
            qp = q_expr.subs(x, r + eps)
            if qm * qp < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
            
    B = max(S(0), max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def construct():
    """Return 12 double-root locations."""
    deadline = time.time() + 580.0 # 50s safety margin
    
    # 1. Generate base nodes: Roots of L_12^(-1/2)(x)
    x = sp.symbols('x')
    alpha = sp.Rational(-1, 2)
    L12 = genlaguerre(12, alpha)(x)
    try:
        roots = sp.nroots(L12, n=15)
        z_base = [float(r) for r in roots if r.is_real and r > 0]
        if len(z_base) != 12:
            raise ValueError("Did not find 12 positive real roots")
        z_base.sort()
    except Exception:
        # Fallback to Bessel squares if Laguerre roots fail
        z_base = [2.4048**2, 5.5200**2, 8.6537**2, 11.7915**2, 14.9309**2, 18.0710**2,
                  21.2116**2, 24.3524**2, 27.4934**2, 30.6346**2, 33.7758**2, 36.9171**2]

    best_score = float('inf')
    best_z = z_base
    
    # 2. Evaluate base score
    if time.time() < deadline - 60:
        score = _compute_score_exact(z_base)
        if score is not None:
            best_score = score
            best_z = list(z_base)

    # 3. Global scaling search
    scales = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    for s in scales:
        if time.time() > deadline - 120:
            break
        candidate = [z * s for z in z_base]
        score = _compute_score_exact(candidate)
        if score is not None and score < best_score:
            best_score = score
            best_z = candidate

    # 4. Local refinement around best_z
    # We will try to perturb each z_i slightly to see if it reduces the max root.
    # This is expensive, so we limit the number of perturbations.
    perturbations = [0.9, 1.1, 0.8, 1.2]
    
    # Iterate a few times to allow for sequential improvements
    for _iter in range(3):
        if time.time() > deadline - 120:
            break
        improved = False
        for i in range(12):
            if time.time() > deadline - 120:
                break
            for p in perturbations:
                if time.time() > deadline - 120:
                    break
                candidate = list(best_z)
                candidate[i] = best_z[i] * p
                # Check validity: distinct and positive
                valid = True
                for j in range(12):
                    if candidate[j] <= 0:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(12):
                    for k in range(j+1, 12):
                        if abs(candidate[j] - candidate[k]) < 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                score = _compute_score_exact(candidate)
                if score is not None and score < best_score:
                    best_score = score
                    best_z = candidate
                    improved = True
                    # Break out of inner loops to restart refinement from new best
                    break
            if improved:
                break
        if not improved:
            break

    return best_z

# EVOLVE-BLOCK-END
