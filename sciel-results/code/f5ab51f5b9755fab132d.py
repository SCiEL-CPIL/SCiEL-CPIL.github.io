import sympy as sp
import numpy as np
import time
import math
import mpmath as mp

def _gen_laguerre(n, alpha, x_val):
    """Compute L_n^{(alpha)}(x) using explicit sum (for mpmath)."""
    total = mp.mpf(0)
    for j in range(n + 1):
        total += ((-1) ** j) * mp.binomial(n + alpha, n - j) * (x_val ** j) / mp.factorial(j)
    return total

def _gen_laguerre_deriv(n, alpha, x_val):
    """d/dx L_n^{(alpha)}(x) = -L_{n-1}^{(alpha+1)}(x)."""
    if n == 0:
        return mp.mpf(0)
    return -_gen_laguerre(n - 1, alpha + 1, x_val)

def evaluate_candidate_exact(z_list):
    """
    Exact sympy evaluation. Returns B (float) or None if invalid.
    """
    try:
        z_rat = [sp.Rational(float(z)) for z in z_list]
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)

        Ls = [sp.Laguerre(2 * k, alpha, x) for k in range(26)]

        M = sp.zeros(26, 26)
        rhs = sp.zeros(26, 1)

        # g(0) = 0
        for k in range(26):
            M[0, k] = Ls[k].subs(x, 0)
        rhs[0] = 0

        # g'(0) = 1
        for k in range(26):
            M[1, k] = sp.diff(Ls[k], x).subs(x, 0)
        rhs[1] = 1

        idx = 2
        for z in z_rat:
            for k in range(26):
                M[idx, k] = Ls[k].subs(x, z)
            rhs[idx] = 0
            idx += 1
            for k in range(26):
                M[idx, k] = sp.diff(Ls[k], x).subs(x, z)
            rhs[idx] = 0
            idx += 1

        try:
            sol = M.LUsolve(rhs)
        except Exception:
            return None

        cs = sp.symbols('c0:26')
        g = sum(cs[k] * Ls[k] for k in range(26))
        g_sol = g.subs(list(zip(cs, sol)))

        denom = x
        for z in z_rat:
            denom *= (x - z) ** 2

        q, rem = sp.div(g_sol, denom)
        if rem != 0:
            return None

        try:
            roots = sp.real_roots(q)
        except Exception:
            return None

        if not roots:
            return 0.0

        max_r = 0.0
        for r in roots:
            try:
                r_float = float(r)
            except Exception:
                continue
            if not np.isfinite(r_float) or r_float <= 0:
                continue
            delta = 1e-198
            try:
                v_minus = float(q.subs(x, r_float - delta))
                v_plus = float(q.subs(x, r_float + delta))
            except Exception:
                continue
            if v_minus * v_plus < 0:
                if r_float > max_r:
                    max_r = r_float
        return float(max_r)
    except Exception:
        return None

def evaluate_candidate_fast(z_list):
    """
    Fast mpmath evaluation. Returns B estimate or large penalty.
    Uses high precision and robust root finding.
    """
    try:
        z = sorted([float(zz) for zz in z_list])
        if len(z) != 12:
            return 1e10
        if any(zi <= 0 for zi in z):
            return 1e10
        for i in range(len(z) - 1):
            if z[i + 1] - z[i] < 1e-4:
                return 1e10

        mp.mp.dps = 50
        alpha = mp.mpf(-1) / 2

        # Precompute Laguerre polynomials coefficients for faster eval?
        # Or just use the sum formula.
        
        # Build system matrix
        M = mp.matrix(26, 26)
        rhs = mp.matrix(26, 1)

        for k in range(26):
            M[0, k] = _gen_laguerre(2 * k, alpha, mp.mpf(0))
        rhs[0] = 0

        for k in range(26):
            M[1, k] = _gen_laguerre_deriv(2 * k, alpha, mp.mpf(0))
        rhs[1] = 1

        idx = 2
        for zz in z:
            zv = mp.mpf(zz)
            for k in range(26):
                M[idx, k] = _gen_laguerre(2 * k, alpha, zv)
            rhs[idx] = 0
            idx += 1
            for k in range(26):
                M[idx, k] = _gen_laguerre_deriv(2 * k, alpha, zv)
            rhs[idx] = 0
            idx += 1

        try:
            sol = mp.lu_solve(M, rhs)
        except Exception:
            return 1e10

        def g_val(xv):
            total = mp.mpf(0)
            for k in range(26):
                total += sol[k] * _gen_laguerre(2 * k, alpha, xv)
            return total

        def denom_val(xv):
            d = xv
            for zz in z:
                d *= (xv - mp.mpf(zz)) ** 2
            return d

        def q_val(xv):
            return g_val(xv) / denom_val(xv)

        # Scan for roots of q(x)
        # q(x) is a polynomial of degree 50 - 25 = 25? No.
        # deg(g) = 50. deg(denom) = 1 + 12*2 = 25.
        # So q(x) is a polynomial of degree 25.
        
        # We want the largest real root.
        # Let's scan from a high value downwards.
        # How high can the largest root be?
        # The roots of q are related to the asymptotic behavior.
        # Let's try to find all real roots using a robust method.
        
        # Since q is a polynomial of degree 25, it has at most 25 real roots.
        # We can find them by scanning for sign changes.
        
        max_z = max(z)
        # The largest root is likely not too far from max_z.
        # Let's scan from max_z * 0.1 to max_z * 10 + 100
        lower_bound = max_z * 0.1
        upper_bound = max_z * 5 + 50
        
        roots_found = []
        
        # Coarse scan
        N_scan = 500
        xs = np.linspace(lower_bound, upper_bound, N_scan)
        
        prev_x = None
        prev_q = None
        
        for xi in xs:
            if xi <= 0: continue
            xv = mp.mpf(xi)
            # Avoid singularities (z_i and 0)
            if any(abs(xi - zz) < 1e-6 for zz in z) or xi < 1e-6:
                prev_x = None
                prev_q = None
                continue
                
            try:
                qi = q_val(xv)
            except Exception:
                prev_x = None
                prev_q = None
                continue
                
            if prev_q is not None and qi != 0:
                if prev_q * qi < 0:
                    # Sign change between prev_x and xi
                    a = mp.mpf(prev_x)
                    b = mp.mpf(xi)
                    try:
                        # Use findroot with bracketing
                        r = mp.findroot(lambda t: q_val(t), (a, b))
                        r_f = float(r)
                        if 0 < r_f:
                            # Check if it's a real root (not a pole, though q is polynomial)
                            # Verify it's a root
                            if abs(q_val(mp.mpf(r_f))) < 1e-20:
                                roots_found.append(r_f)
                    except Exception:
                        pass
            
            prev_x = xi
            prev_q = qi

        if not roots_found:
            # Maybe the largest root is smaller than lower_bound?
            # Or maybe we missed it.
            # Let's try a finer scan or different range.
            # But for now, if we found none, return a penalty or 0?
            # If q has no real roots > 0, B=0.
            return 0.0

        return max(roots_found)
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # Initial guess: Bessel zeros squared
    j0_zeros = [
        2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
        14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
        27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
    ]
    bessel_sq = [z * z for z in j0_zeros]
    
    best_B = float('inf')
    best_z = None
    
    # 1. Evaluate the Bessel baseline exactly to have a valid fallback
    B_base = evaluate_candidate_exact(bessel_sq)
    if B_base is not None:
        best_B = B_base
        best_z = bessel_sq.copy()
    else:
        # Fallback if Bessel is invalid (shouldn't happen)
        best_z = bessel_sq.copy()
        best_B = 1e10

    # 2. Global Search using Fast Evaluator
    # We will use a combination of random sampling and local refinement.
    # The search space is z_1 < z_2 < ... < z_12.
    
    def fast_obj(z):
        return evaluate_candidate_fast(z)
    
    # Generate diverse initial candidates
    candidates = [bessel_sq.copy()]
    
    # Random perturbations of Bessel
    np.random.seed(42)
    for _ in range(20):
        perturbed = [z * (1 + np.random.normal(0, 0.2)) for z in bessel_sq]
        perturbed = sorted(perturbed)
        if all(z > 0 for z in perturbed) and all(perturbed[i+1] - perturbed[i] > 1e-3 for i in range(11)):
            candidates.append(perturbed)
            
    # Uniform random in a box
    for _ in range(20):
        # z_i roughly in [1, 100]
        z_rand = sorted(np.random.uniform(1, 100, 12))
        if all(z_rand[i+1] - z_rand[i] > 1e-3 for i in range(11)):
            candidates.append(z_rand)
            
    # Evaluate candidates with fast evaluator
    for zc in candidates:
        if time.time() > deadline - 100:
            break
        B_fast = fast_obj(zc)
        if B_fast < best_B:
            # Verify with exact evaluator
            B_exact = evaluate_candidate_exact(zc)
            if B_exact is not None and B_exact < best_B:
                best_B = B_exact
                best_z = zc.copy()

    # 3. Local Refinement using Nelder-Mead on Fast Objective
    # We refine the best candidate found so far.
    # We can also try to refine a few other good candidates.
    
    from scipy.optimize import minimize
    
    # Sort candidates by fast score to pick good starting points
    scored_candidates = []
    for zc in candidates:
        if time.time() > deadline - 200:
            break
        B_fast = fast_obj(zc)
        scored_candidates.append((B_fast, zc))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 5 distinct candidates for refinement
    refined_starts = []
    for score, zc in scored_candidates:
        if len(refined_starts) >= 5:
            break
        # Check if distinct from existing starts
        is_distinct = True
        for existing in refined_starts:
            if np.allclose(zc, existing, atol=1e-3):
                is_distinct = False
                break
        if is_distinct:
            refined_starts.append(zc)
            
    for z_start in refined_starts:
        if time.time() > deadline - 120:
            break
        
        def obj(z):
            return fast_obj(z)
            
        try:
            # Use Nelder-Mead
            # Bounds are tricky for NM, so we rely on the penalty in fast_obj
            res = minimize(obj, z_start, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if res.fun < best_B:
                z_opt = sorted(res.x)
                if all(zi > 0 for zi in z_opt) and len(set(round(zi, 8) for zi in z_opt)) == 12:
                    B_exact = evaluate_candidate_exact(z_opt)
                    if B_exact is not None and B_exact < best_B:
                        best_B = B_exact
                        best_z = z_opt.copy()
        except Exception:
            pass

    # 4. Final Polish: Try a few more random starts near the best solution
    if best_z is not None:
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            z_perturb = [z * (1 + np.random.normal(0, 0.05)) for z in best_z]
            z_perturb = sorted(z_perturb)
            if all(z > 0 for z in z_perturb) and all(z_perturb[i+1] - z_perturb[i] > 1e-4 for i in range(11)):
                B_fast = fast_obj(z_perturb)
                if B_fast < best_B:
                    B_exact = evaluate_candidate_exact(z_perturb)
                    if B_exact is not None and B_exact < best_B:
                        best_B = B_exact
                        best_z = z_perturb.copy()

    return best_z
