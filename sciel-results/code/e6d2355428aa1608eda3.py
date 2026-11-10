import time
import sys

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh

def construct():
    """
    Construct 12 double-root locations z_1..z_12.
    
    Strategy: We want to minimize B, which is the largest real root r of q(x) = g(x)/(x * prod (x-z_i)^2)
    such that q changes sign at r. B = max(0, largest counted r).
    
    The key insight: B is determined by the behavior of the rational function q(x).
    Since g(x) is a linear combination of even-degree Laguerre polynomials L_{2k}^{(-1/2)}(x) for k=0..25,
    and g(x) has double zeros at each z_i plus a simple zero at x=0 (from the x factor in denominator),
    we need to position the z_i to push the largest positive real root of q as close to 0 as possible.
    
    A practical approach: Use optimization to find z_i that minimize an upper bound on B.
    
    Key observations:
    1. The Laguerre polynomials L_n^{(-1/2)}(x) have all roots in (0, infinity) for n >= 1.
    2. The even-degree ones L_{2k}^{(-1/2)}(x) have 2k roots, all positive.
    3. g(x) is constructed to have specific interpolation properties at the z_i.
    
    Let's think about what controls B. The function q(x) = g(x)/(x * prod_i (x-z_i)^2) is a polynomial
    of degree at most 50 - 1 - 24 = 25. Its roots are determined by the coefficients c_k which are
    solved from the linear system.
    
    Alternative approach: Directly optimize the z_i to minimize B using a proxy objective.
    
    Since computing B exactly requires solving the full linear system and finding roots of q,
    which is expensive, let's use a heuristic:
    
    The roots of generalized Laguerre polynomials L_n^{alpha}(x) are well-studied. For alpha = -1/2,
    the roots of L_n^{(-1/2)}(x) are related to the zeros of certain Bessel functions.
    
    Let's try placing the z_i near the roots of a specific Laguerre polynomial to leverage orthogonality.
    
    Actually, let's think more carefully. The system is:
    g(x) = sum_{k=0}^{25} c_k L_{2k}^{(-1/2)}(x)
    g(0) = 0
    g'(0) = 1
    g(z_i) = 0 for i=1..12
    g'(z_i) = 0 for i=1..12
    
    That's 2 + 12 + 12 = 26 equations for 26 unknowns c_k.
    
    The function q(x) = g(x) / (x * prod_i (x-z_i)^2) is a polynomial of degree 25.
    We want to minimize the largest positive real root of q.
    
    Let's use a numerical optimization approach where we:
    1. Parameterize z_i as positive distinct values
    2. Use a proxy objective that correlates with B
    3. Optimize using scipy
    
    For the proxy, note that if we can make g(x) small for x > some threshold, then q(x) will be small there.
    But this is hard to control directly.
    
    Alternative: Use the fact that the roots of q are related to the eigenvalues of a certain matrix.
    
    Let me try a simpler heuristic first: place z_i at the roots of L_{24}^{(-1/2)}(x), which has 24 roots.
    We need 12 of them. Let's take every other root or the 12 largest/smallest.
    
    Actually, let's just try to optimize numerically with a time budget.
    """
    start_time = time.time()
    deadline = start_time + 550  # Leave 80s margin before 630s kill
    
    # Initial guess: spread z_i over a reasonable range
    # Laguerre roots for L_n^{(-1/2)} are roughly in (0, ~4n) for large n
    # Let's start with z_i spread from 0.5 to 50
    
    def compute_B(z):
        """Compute B for given z locations. Returns B or inf if invalid."""
        if time.time() > deadline:
            return float('inf')
        
        import sympy as sp
        
        # Convert z to rational
        z_rational = [sp.Rational(float(zi)) for zi in z]
        
        # Check distinctness and positivity
        z_sorted = sorted(z_rational)
        for i in range(len(z_sorted) - 1):
            if z_sorted[i] <= 0 or z_sorted[i+1] - z_sorted[i] < sp.Rational(1, 10**10):
                return float('inf')
        
        # Build the linear system
        x = sp.symbols('x')
        alpha = sp.Rational(-1, 2)
        
        # Get Laguerre polynomials L_{2k}^{(-1/2)}(x) for k=0..25
        L = []
        for k in range(26):
            L.append(sp.laguerre(2*k, alpha, x))
        
        c = sp.symbols('c0:26')
        g = sum(c[k] * L[k] for k in range(26))
        
        # Equations:
        # 1. g(0) = 0
        eqs = [sp.Eq(g.subs(x, 0), 0)]
        
        # 2. g'(0) = 1
        g_prime = sp.diff(g, x)
        eqs.append(sp.Eq(g_prime.subs(x, 0), 1))
        
        # 3. g(z_i) = 0 and g'(z_i) = 0 for each i
        for zi in z_rational:
            eqs.append(sp.Eq(g.subs(x, zi), 0))
            eqs.append(sp.Eq(g_prime.subs(x, zi), 0))
        
        # Solve the linear system
        # Extract coefficients
        A = []
        b = []
        for eq in eqs:
            # eq is of form sum_j A_j * c_j = b_val
            expr = sp.expand(eq.lhs - eq.rhs)
            # expr = sum_j coeff_j * c_j
            coeff_vec = [sp.expand(expr.coeff(c[j])) for j in range(26)]
            A.append(coeff_vec)
            b.append(eq.rhs - eq.lhs)  # This should be 0 for all except possibly... wait
        
        # Actually, let's redo this more carefully
        # Each equation is: sum_j A_{ij} c_j = b_i
        # where b_i is the constant term
        
        A_mat = sp.Matrix(26, 26, lambda i, j: 0)
        b_vec = sp.Matrix(26, 1)
        
        for i, eq in enumerate(eqs):
            lhs = sp.expand(eq.lhs)
            rhs = sp.expand(eq.rhs)
            for j in range(26):
                A_mat[i, j] = lhs.coeff(c[j])
            b_vec[i] = rhs - lhs.subs({c[j]: 0 for j in range(26)})
        
        # Check if system is singular
        if A_mat.det() == 0:
            return float('inf')
        
        # Solve
        try:
            sol = A_mat.LUsolve(b_vec)
        except Exception:
            return float('inf')
        
        # Build g with solved coefficients
        g_sol = sum(sol[j] * L[j] for j in range(26))
        
        # Build denominator: x * prod_i (x - z_i)^2
        denom = x
        for zi in z_rational:
            denom *= (x - zi)**2
        
        # q(x) = g_sol / denom, should be a polynomial
        q = sp.cancel(g_sol / denom)
        
        # If q is not a polynomial, something went wrong
        if not q.is_polynomial():
            return float('inf')
        
        # Find real roots of q
        try:
            real_roots = sp.real_roots(q)
        except Exception:
            return float('inf')
        
        # Count roots where q has opposite signs at r - 1e-198 and r + 1e-198
        counted_roots = []
        for r in real_roots:
            r_float = float(r)
            if r_float <= 0:
                continue
            
            # Evaluate q at r - 1e-198 and r + 1e-198
            # Use high precision
            eps = sp.Rational(1, 10**198)
            q_left = q.subs(x, r - eps)
            q_right = q.subs(x, r + eps)
            
            # Check if signs are opposite
            sign_left = 1 if q_left > 0 else (-1 if q_left < 0 else 0)
            sign_right = 1 if q_right > 0 else (-1 if q_right < 0 else 0)
            
            if sign_left != 0 and sign_right != 0 and sign_left * sign_right < 0:
                counted_roots.append(r_float)
        
        if not counted_roots:
            B = 0.0
        else:
            B = max(counted_roots)
        
        return B
    
    # Initial guess
    z0 = np.linspace(1.0, 30.0, 12)
    
    # Objective function
    def objective(z):
        # Ensure z are positive and distinct
        z = np.sort(z)
        if any(z <= 0) or any(np.diff(z) < 1e-6):
            return 1e10
        return compute_B(z)
    
    # Try to optimize
    best_z = z0.copy()
    best_B = objective(z0)
    
    if time.time() < deadline - 100:
        # Use Nelder-Mead optimization
        try:
            result = minimize(objective, z0, method='Nelder-Mead', 
                             options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4,
                                      'adaptive': True})
            if result.success and result.fun < best_B:
                best_z = result.x
                best_B = result.fun
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 50:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            z_rand = np.sort(np.random.uniform(0.5, 40.0, 12))
            B_rand = objective(z_rand)
            if B_rand < best_B:
                best_z = z_rand
                best_B = B_rand
    
    # Ensure output is valid
    result = np.sort(best_z)
    if any(result <= 0) or any(np.diff(result) < 1e-8):
        # Fallback to initial guess
        result = z0
    
    return list(result)
# EVOLVE-BLOCK-END
