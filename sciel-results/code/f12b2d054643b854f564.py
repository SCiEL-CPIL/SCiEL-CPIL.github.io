import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize
import time

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 30.0  # Keep construct() fast, evaluator has 600s total
    
    n_roots = 12
    max_k = 25  # Degrees 0, 2, ..., 50
    
    # Precompute Laguerre polynomials values and derivatives at 0
    # L_n^(alpha)(x)
    alpha = -0.5
    
    # We need L_{2k}(0) and L'_{2k}(0) for k=0..25
    # L_n(0) = binom(n+alpha, n)
    # L_n'(x) = -L_{n-1}^{(alpha+1)}(x)
    
    # Let's build the matrix A for the linear system
    # Variables: c_0, ..., c_25
    # Equations:
    # 1. g(0) = 0
    # 2. g'(0) = 1
    # 3..14. g(z_i) = 0 for i=1..12
    # 15..26. g'(z_i) = 0 for i=1..12
    
    def get_laguerre_coeffs(z):
        """
        Returns A (26x26) and b (26,) for the system A*c = b
        z is an array of 12 floats
        """
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        for k in range(26):
            n = 2 * k
            A[0, k] = eval_genlaguerre(n, alpha, 0.0)
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        for k in range(26):
            n = 2 * k
            if n == 0:
                deriv = 0.0
            else:
                # L_n'(x) = -L_{n-1}^{(alpha+1)}(x)
                deriv = -eval_genlaguerre(n-1, alpha+1, 0.0)
            A[1, k] = deriv
        b[1] = 1.0
        
        # Rows 2..13: g(z_i) = 0
        for i in range(n_roots):
            zi = z[i]
            for k in range(26):
                n = 2 * k
                A[2 + i, k] = eval_genlaguerre(n, alpha, zi)
            b[2 + i] = 0.0
            
        # Rows 14..25: g'(z_i) = 0
        for i in range(n_roots):
            zi = z[i]
            for k in range(26):
                n = 2 * k
                if n == 0:
                    deriv = 0.0
                else:
                    deriv = -eval_genlaguerre(n-1, alpha+1, zi)
                A[14 + i, k] = deriv
            b[14 + i] = 0.0
            
        return A, b

    def compute_B_approx(z):
        """
        Computes the approximate B value for a given set of z.
        Returns a large number if invalid.
        """
        try:
            A, b = get_laguerre_coeffs(z)
            c = np.linalg.solve(A, b)
            
            # Construct g(x) and g'(x) evaluation functions
            def g(x):
                return np.sum([c[k] * eval_genlaguerre(2*k, alpha, x) for k in range(26)])
            
            def gp(x):
                s = 0.0
                for k in range(26):
                    n = 2*k
                    if n > 0:
                        s += c[k] * (-eval_genlaguerre(n-1, alpha+1, x))
                return s

            # We need to find roots of q(x) = g(x) / D(x)
            # q(x) is a polynomial of degree 50 - (1 + 2*12) = 25.
            # Instead of symbolic division, we can find roots of g(x) 
            # that are not roots of D(x).
            # But g(x) has roots at z_i (double) and 0? 
            # g(0)=0, g'(0)=1 so 0 is a simple root of g.
            # g(z_i)=0, g'(z_i)=0 so z_i are double roots of g.
            # Total roots accounted for: 1 (at 0) + 12*2 (at z_i) = 25 roots.
            # Degree of g is 50. So there are 25 other roots.
            # q(x) will have these 25 roots.
            
            # Find all roots of g(x)
            # We can use numpy roots on the coefficients of g in the Laguerre basis?
            # No, convert to monomial basis or just use a root finder on the interval.
            
            # Better: Convert g(x) to monomial basis?
            # L_n(x) = sum_{j=0}^n binom(n, j) (-1)^j / j! x^j
            # This is messy.
            
            # Alternative: Use scipy.optimize.root to find roots in intervals.
            # Or, since we just need the LARGEST real root of q(x),
            # and q(x) is g(x)/D(x), the roots of q are the roots of g
            # excluding 0 and z_i.
            
            # Let's find roots of g(x) in a large interval [0, 200].
            # We know roots are at 0, z_1..z_12.
            # We need the other 25 roots.
            
            # Use a simple bisection or root finding on g(x).
            # Since g is oscillatory, we can scan.
            
            # Actually, we can just evaluate the "largest root" by checking
            # where g(x) crosses zero, excluding the known roots.
            
            # Let's find all real roots of g(x) using a grid search + refinement.
            # Range: 0 to 100 (Laguerre roots grow with n, max n=50, root ~ 4*50=200? 
            # Actually L_n^(alpha) largest root ~ 4n + 2alpha + 1. For n=50, alpha=-0.5, ~ 200 - 1 = 199.
            
            xs = np.linspace(0.01, 200, 10000)
            ys = np.array([g(x) for x in xs])
            
            roots = []
            for i in range(len(xs)-1):
                if ys[i] == 0:
                    roots.append(xs[i])
                elif ys[i]*ys[i+1] < 0:
                    # Bisection
                    a, b = xs[i], xs[i+1]
                    fa, fb = ys[i], ys[i+1]
                    for _ in range(50):
                        m = (a+b)/2
                        fm = g(m)
                        if fa*fm <= 0:
                            b, fb = m, fm
                        else:
                            a, fa = m, fm
                    roots.append((a+b)/2)
            
            # Filter roots: exclude 0 and z_i (within tolerance)
            valid_roots = []
            for r in roots:
                is_zero = abs(r) < 1e-6
                is_zi = any(abs(r - zi) < 1e-5 for zi in z)
                if not is_zero and not is_zi and r > 0:
                    valid_roots.append(r)
            
            if not valid_roots:
                return 1e10 # Invalid or no roots found
            
            B = max(valid_roots)
            return B
            
        except Exception:
            return 1e10

    # Initial guess: Roots of L_12^{-1/2}(x)
    # We can approximate these or just use a simple grid
    # Let's use a simple spread: 1, 2, ..., 12
    z0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    
    # Optimize
    # Use Powell method which doesn't need gradients
    # Constraints: z_i > 0, distinct.
    # We can penalize violations in the objective function.
    
    def objective(z):
        # Penalize non-positive or close values
        penalty = 0
        for i in range(len(z)):
            if z[i] < 0.1:
                penalty += (0.1 - z[i])**2 * 1000
            for j in range(i+1, len(z)):
                if abs(z[i] - z[j]) < 0.1:
                    penalty += (0.1 - abs(z[i] - z[j]))**2 * 1000
        return compute_B_approx(z) + penalty

    # Run optimization
    # Limit iterations to stay within time
    result = None
    try:
        # Use differential evolution for global search? Might be slow.
        # Powell is local. Let's try a few random starts with Powell.
        best_z = z0
        best_val = objective(z0)
        
        # Try a few random perturbations
        for _ in range(5):
            if time.time() - start_time > deadline - 5:
                break
            z_rand = np.sort(z0 + np.random.normal(0, 1, 12))
            z_rand[z_rand < 0.5] = 0.5 + np.random.rand(12)
            res = minimize(objective, z_rand, method='Powell', options={'maxiter': 20, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_z = res.x
                
    except Exception:
        pass

    # Ensure final result is valid
    if best_z is None:
        best_z = z0

    # Sort and return
    best_z = np.sort(best_z)
    return list(best_z)
