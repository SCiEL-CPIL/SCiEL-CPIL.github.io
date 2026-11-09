import time
import math
import sympy
from sympy import Rational, symbols, expand, LUsolve, Matrix, genlaguerre, real_roots, div

def construct():
    """
    Returns 12 double-root locations z_1..z_12 (list of floats).
    """
    t_start = time.time()
    DEADLINE = 600
    SAFETY_MARGIN = 30
    
    # Candidate sets of z_i
    # Heuristic: Smaller z_i tends to push roots of q(x) to smaller values or negative.
    candidates = [
        [i * 1e-3 for i in range(1, 13)],       # Very small
        [i * 1e-2 for i in range(1, 13)],       # Small
        [i * 0.05 for i in range(1, 13)],       # Medium-small
        [i * 0.1 for i in range(1, 13)],        # Medium
        [i * 0.5 for i in range(1, 13)],        # Larger
        [i * 1.0 for i in range(1, 13)],        # Large
        [4.0 * (i + 1) for i in range(12)],     # Original seed
    ]
    
    best_z = None
    best_score = float('inf')
    best_b = None
    
    # Pre-compute Laguerre polynomials for speed
    # We need L_{2k}^{(-1/2)}(x) and L_{2k-1}^{(1/2)}(x) for k=0..25
    x = symbols('x')
    alpha_g = Rational(-1, 2)
    alpha_d = Rational(1, 2)
    
    P_k = [] # L_{2k}^{(-1/2)}(x)
    Q_k = [] # L_{2k-1}^{(1/2)}(x)
    
    for k in range(26):
        P_k.append(genlaguerre(2*k, alpha_g, x))
        if k == 0:
            Q_k.append(Rational(0)) # Derivative of L_0 is 0
        else:
            # d/dx L_n^(alpha) = -L_{n-1}^{(alpha+1)}
            # Here n=2k, alpha=-1/2. So derivative is -L_{2k-1}^{(1/2)}
            Q_k.append(-genlaguerre(2*k - 1, alpha_d, x))
            
    # Note: g'(x) = sum c_k * d/dx(L_{2k}^{(-1/2)}(x)) = sum c_k * Q_k(x)
    # Wait, the identity is d/dx L_n^(alpha) = -L_{n-1}^{(alpha+1)}.
    # So d/dx L_{2k}^{(-1/2)} = -L_{2k-1}^{(1/2)}.
    # My Q_k above is defined as -genlaguerre(...). So g'(x) = sum c_k * Q_k(x) is correct.
    
    for z_list in candidates:
        if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
            break
            
        try:
            b_val = _evaluate_with_precomp(z_list, P_k, Q_k, x)
            if b_val is not None:
                score = b_val / (2 * math.pi)
                if score < best_score:
                    best_score = score
                    best_b = b_val
                    best_z = z_list
        except Exception:
            continue
            
    if best_z is None:
        # Fallback to seed if all fail
        return [4.0 * (i + 1) for i in range(12)]
        
    return [float(zi) for zi in best_z]

def _evaluate_with_precomp(z_list, P_k, Q_k, x):
    """
    Evaluates B for a given list of z values using pre-computed Laguerre polynomials.
    Returns B (float) or None if invalid.
    """
    # Convert z to rationals
    z_rat = [Rational(float(zi)) for zi in z_list]
    
    # Check distinctness and positivity
    if len(set(z_rat)) != 12:
        return None
    if any(zi <= 0 for zi in z_rat):
        return None
    
    # Build system A * c = b
    # Unknowns: c_0 ... c_25
    # Constraints:
    # 1. g(0) = sum c_k P_k(0) = 0
    # 2. g'(0) = sum c_k Q_k(0) = 1
    # 3. g(z_i) = sum c_k P_k(z_i) = 0 for i=1..12
    # 4. g'(z_i) = sum c_k Q_k(z_i) = 0 for i=1..12
    
    A = Matrix(26, 26, 0)
    b = Matrix(26, 1, 0)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = P_k[k].subs(x, 0)
    b[0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = Q_k[k].subs(x, 0)
    b[1] = 1
    
    # Rows 2..13: g(z_i) = 0
    for i in range(12):
        row_idx = 2 + i
        zi = z_rat[i]
        for k in range(26):
            A[row_idx, k] = P_k[k].subs(x, zi)
        b[row_idx] = 0
        
    # Rows 14..25: g'(z_i) = 0
    for i in range(12):
        row_idx = 14 + i
        zi = z_rat[i]
        for k in range(26):
            A[row_idx, k] = Q_k[k].subs(x, zi)
        b[row_idx] = 0
        
    # Check singularity
    det_A = A.det()
    if det_A == 0:
        return None
        
    # Solve
    c_sol = A.LUsolve(b)
    
    # Construct g(x)
    g = sum(c_sol[k] * P_k[k] for k in range(26))
    g = expand(g)
    
    # Construct denominator D(x) = x * prod (x - z_i)^2
    D = x
    for zi in z_rat:
        D = D * (x - zi)**2
    D = expand(D)
    
    # Polynomial division
    q, rem = div(g, D, x)
    
    if rem != 0:
        return None
        
    # Find real roots of q
    # q is a polynomial of degree 25
    real_roots = real_roots(q)
    
    if not real_roots:
        return 0.0
        
    # Check sign change at r +/- 1e-198
    eps = Rational(1, 10)**198
    counted_roots = []
    for r in real_roots:
        # r is a RootOf object or Rational/AlgebraicNumber
        # We need to evaluate q at r-eps and r+eps
        # sympy's real_roots returns exact roots.
        # Substituting into q might be slow.
        # Alternative: check multiplicity? 
        # The condition is "strictly opposite signs". This implies simple roots with sign change.
        # For real polynomials, simple roots always have sign changes.
        # Roots with even multiplicity do not.
        # So we just need to check if the root is simple?
        # Actually, the prompt says: "counts if q has strictly opposite signs at r - 1e-198 and r + 1e-198".
        # This is effectively checking if the root is of odd multiplicity and the function crosses zero.
        # Since coefficients are real, any simple real root has a sign change.
        # Multiple roots (even) do not.
        # So we can just check if the root is simple.
        # However, evaluating q(r-eps) is robust.
        
        # To save time, we can assume simple roots are the ones we care about.
        # Let's just collect all real roots first, then filter if time permits.
        # Given the time constraint, let's assume all real roots found by real_roots are candidates
        # and verify sign change only for the largest few?
        # No, B is the max. So we only care about the largest root that qualifies.
        # Let's sort roots in descending order.
        
        pass
        
    # Sort roots in descending order
    # real_roots returns a list of RootOf objects.
    # We need to compare them.
    # sympy's real_roots returns them in ascending order? Not guaranteed.
    # Let's convert to floats for sorting, then re-evaluate exactness for the top candidate.
    
    root_floats = []
    for r in real_roots:
        try:
            root_floats.append((float(r), r))
        except:
            continue
            
    root_floats.sort(reverse=True)
    
    # Check sign change for the largest roots until we find one that counts
    # If the largest root doesn't count, check the next, etc.
    # But B is max(0, largest counted r).
    # If no root counts, B=0.
    
    for r_float, r_exact in root_floats:
        if r_float <= 0:
            # Since we want max(0, ...), if all remaining are <= 0, B=0
            # And since we are checking in descending order, if current is <=0, all subsequent are <=0.
            # So B=0.
            break
            
        # Check sign change
        val_left = q.subs(x, r_exact - eps)
        val_right = q.subs(x, r_exact + eps)
        
        if val_left * val_right < 0:
            # This root counts
            return float(r_exact)
            
    return 0.0

if __name__ == "__main__":
    z = construct()
    print(f"Constructed z: {z}")
