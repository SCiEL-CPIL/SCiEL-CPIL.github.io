import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre
from scipy.optimize import minimize_scalar

def construct():
    # Get roots of L_12^(-1/2)(x)
    # These are positive and distinct.
    p = genlaguerre(12, -0.5)
    roots_base = p.roots
    # Ensure they are sorted and real
    roots_base = sorted([r.real for r in roots_base if abs(r.imag) < 1e-10])
    if len(roots_base) != 12:
        # Fallback to seed if something went wrong with root finding
        return [4.0 * (i + 1) for i in range(12)]

    def objective(lam):
        if lam <= 0:
            return 1e10
        zs = np.array(roots_base) * lam
        
        # Check distinctness and positivity
        # Since roots_base are distinct and lam > 0, they are distinct.
        if np.any(zs <= 0):
            return 1e10
            
        alpha = -0.5
        n_max = 50 # Degree 50, so k=0..25, 2k=0..50
        
        # Build system A c = b
        # 26 equations, 26 unknowns
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        for k in range(26):
            A[0, k] = eval_genlaguerre(2*k, alpha, 0)
        b[0] = 0
        
        # Row 1: g'(0) = 1
        for k in range(26):
            if k == 0:
                A[1, k] = 0
            else:
                A[1, k] = -eval_genlaguerre(2*k-1, alpha+1, 0)
        b[1] = 1
        
        # Rows for z_i
        for i, z in enumerate(zs):
            row_g = 2 + i
            row_gp = 14 + i
            
            for k in range(26):
                A[row_g, k] = eval_genlaguerre(2*k, alpha, z)
                if k == 0:
                    A[row_gp, k] = 0
                else:
                    A[row_gp, k] = -eval_genlaguerre(2*k-1, alpha+1, z)
            
            b[row_g] = 0
            b[row_gp] = 0
        
        # Solve
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e10
        
        # Construct g(x) coefficients
        g_coeffs = np.zeros(51)
        
        for k in range(26):
            p_k = genlaguerre(2*k, alpha, monic=False)
            coeffs = p_k.coeffs
            # coeffs[0] is x^{2k}
            # g_coeffs[idx] is x^{50-idx}
            for j in range(len(coeffs)):
                power = 2*k - j
                idx = 50 - power
                g_coeffs[idx] += c[k] * coeffs[j]
        
        # Construct D(x) = x * prod (x - z_i)^2
        d_poly = np.poly1d([1.0, 0.0]) # x
        for z in zs:
            factor = np.poly1d([1.0, -2*z, z*z])
            d_poly = np.poly1d(d_poly * factor)
        
        # Divide
        g_poly = np.poly1d(g_coeffs)
        q_poly, rem = np.polydiv(g_poly, d_poly)
        
        # Check remainder
        if np.max(np.abs(rem.coeffs)) > 1e-5:
            # If remainder is large, the division is not clean, 
            # but we proceed with the quotient as an approximation
            pass
            
        # Find roots of q
        if len(q_poly.coeffs) <= 1:
            return 0.0
            
        roots = np.roots(q_poly.coeffs)
        
        # Filter real, positive roots
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
        
        if not real_roots:
            return 0.0
            
        return max(real_roots)

    # Optimize lambda
    # Initial guess for lambda: 
    # Seed used z_i = 4, 8, ..., 48.
    # Roots of L_12^-0.5 are roughly in [1, 20]? 
    # Let's estimate. The largest root of L_n is approx 4n + 2alpha + 3. For n=12, alpha=-0.5, approx 48 - 1 + 3 = 50.
    # So the roots are already scaled similarly to the seed's max value.
    # Let's try lambda = 1.0 first.
    
    # Use bounded minimization to ensure lambda stays positive
    # Range [0.1, 10.0] should cover reasonable scales.
    res = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded', options={'xatol': 1e-4})
    
    if res.success:
        lam_opt = res.x
    else:
        lam_opt = 1.0
        
    z_opt = [float(r * lam_opt) for r in roots_base]
    
    # Final check
    if not all(z > 0 for z in z_opt) or len(set(z_opt)) < 12:
        return [4.0 * (i + 1) for i in range(12)]
        
    return z_opt
