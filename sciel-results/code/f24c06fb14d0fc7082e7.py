import time
import numpy as np
import sympy as sp

def construct():
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s kill
    
    # We need to find 12 double-root locations z_1..z_12 > 0, pairwise distinct
    # Strategy: search over structured configurations of z_i
    # The goal is to minimize B = max(0, largest real root of q with sign change)
    # where q(x) = g(x) / (x * prod_i (x-z_i)^2)
    
    # Key insight: if we place z_i at specific locations, we can control where q has roots.
    # A natural approach: try geometric or arithmetic progressions, or optimized placements.
    
    # Let's try a systematic search:
    # 1. Start with simple configurations
    # 2. Use local optimization if time permits
    
    def evaluate_z(z_list):
        """Given 12 z values, compute B."""
        if time.time() > deadline - 30:
            return None  # timeout guard
        
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)
        
        # Build the system: g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
        # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
        # Total: 2 + 2*12 = 26 equations for 26 unknowns c_0..c_25
        
        # Precompute Laguerre polynomials L_{2k}^{alpha}(x) for k=0..25
        # L_n^{alpha}(x) is the generalized Laguerre polynomial
        
        c_vars = sp.symbols('c0:26')
        
        # Build g(x)
        g = sp.zeros()
        for k in range(26):
            n = 2 * k
            L_n = sp.genlaguerre(n, alpha, x)
            g = g + c_vars[k] * L_n
        
        # Compute g(0), g'(0)
        g0 = g.subs(x, 0)
        g_prime = sp.diff(g, x)
        gp0 = g_prime.subs(x, 0)
        
        # Equations
        eqs = [g0, gp0 - 1]
        
        # For each z_i: g(z_i) = 0, g'(z_i) = 0
        z_rationals = []
        for z_val in z_list:
            z_rat = sp.Rational(float(z_val))
            z_rationals.append(z_rat)
            eqs.append(g.subs(x, z_rat))
            eqs.append(g_prime.subs(x, z_rat))
        
        # Solve the linear system
        A = sp.Matrix([[sp.diff(eq, c_vars[j]) for j in range(26)] for eq in eqs])
        b = -sp.Matrix([eq.subs(c_vars, [0]*26) for eq in eqs])
        
        # Actually, let's set up properly: A * c = b
        # g = sum c_k L_{2k}, so g(0) = sum c_k L_{2k}(0), etc.
        # Let me rebuild more carefully
        
        # g(0) = sum_k c_k * L_{2k}(0) = 0
        # g'(0) = sum_k c_k * L'_{2k}(0) = 1
        # g(z_i) = sum_k c_k * L_{2k}(z_i) = 0
        # g'(z_i) = sum_k c_k * L'_{2k}(z_i) = 0
        
        # So the system is: M * c = rhs where M[i][j] = L_{2j}^{(order)}(point_i)
        
        M = sp.Matrix(26, 26, lambda i, j: 0)
        rhs = sp.zeros(26, 1)
        
        # Row 0: g(0) = 0
        for j in range(26):
            L_2j = sp.genlaguerre(2*j, alpha, x)
            M[0, j] = L_2j.subs(x, 0)
        rhs[0] = 0
        
        # Row 1: g'(0) = 1
        for j in range(26):
            L_2j = sp.genlaguerre(2*j, alpha, x)
            L_2j_prime = sp.diff(L_2j, x)
            M[1, j] = L_2j_prime.subs(x, 0)
        rhs[1] = 1
        
        # Rows 2..25: g(z_i)=0, g'(z_i)=0 for i=0..11
        for i in range(12):
            z_i = z_rationals[i]
            # Row 2+2*i: g(z_i) = 0
            for j in range(26):
                L_2j = sp.genlaguerre(2*j, alpha, x)
                M[2+2*i, j] = L_2j.subs(x, z_i)
            rhs[2+2*i] = 0
            
            # Row 3+2*i: g'(z_i) = 0
            for j in range(26):
                L_2j = sp.genlaguerre(2*j, alpha, x)
                L_2j_prime = sp.diff(L_2j, x)
                M[3+2*i, j] = L_2j_prime.subs(x, z_i)
            rhs[3+2*i] = 0
        
        # Solve M * c = rhs
        try:
            c_sol = M.LUsolve(rhs)
        except Exception:
            return None  # singular system
        
        # Build g(x) with solved coefficients
        g_sol = sp.zeros()
        for k in range(26):
            L_2k = sp.genlaguerre(2*k, alpha, x)
            g_sol = g_sol + c_sol[k] * L_2k
        
        # Build denominator: x * prod_i (x - z_i)^2
        denom = x
        for z_i in z_rationals:
            denom = denom * (x - z_i)**2
        
        # q(x) = g(x) / denom
        q = sp.together(g_sol / denom)
        q_poly = sp.Poly(q, x)
        
        # Find real roots of q
        try:
            real_roots = sp.real_roots(q_poly, extension=True)
        except Exception:
            return None
        
        # For each real root r, check sign change at r - 1e-198 and r + 1e-198
        # Since we're working with rationals, use a small rational epsilon
        eps = sp.Rational(1, 10)**198
        
        max_r = 0
        for r in real_roots:
            if r.is_real:
                r_val = r.evalf()
                if r_val > 0:
                    # Check sign change
                    q_at_minus = q.subs(x, r - eps).evalf()
                    q_at_plus = q.subs(x, r + eps).evalf()
                    if q_at_minus * q_at_plus < 0:
                        if r_val > max_r:
                            max_r = r_val
        
        B = max(0, max_r)
        return B
    
    # Search strategy: try different configurations of z_i
    # Configuration 1: arithmetic progression starting from small value
    best_B = float('inf')
    best_z = None
    
    # Try various base configurations
    configs = []
    
    # Arithmetic progressions: z_i = a + i*d for i=0..11
    for a in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for d in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            z_list = [a + i*d for i in range(12)]
            if all(z > 0 for z in z_list):
                configs.append(z_list)
    
    # Geometric progressions: z_i = a * r^i
    for a in [0.1, 0.2, 0.5, 1.0]:
        for r in [1.1, 1.2, 1.3, 1.5, 1.7]:
            z_list = [a * (r ** i) for i in range(12)]
            if all(z > 0 for z in z_list):
                configs.append(z_list)
    
    # Try to evaluate configurations, stopping when time runs low
    for z_list in configs:
        if time.time() > deadline - 60:
            break
        
        B_val = evaluate_z(z_list)
        if B_val is not None and B_val < best_B:
            best_B = B_val
            best_z = z_list[:]
    
    if best_z is not None:
        return best_z
    else:
        # Fallback: return a simple valid configuration
        return [1.0 + i * 0.5 for i in range(12)]
