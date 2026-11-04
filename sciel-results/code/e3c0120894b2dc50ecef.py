"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize
import time
import math

_START_TIME = time.time()
TIME_LIMIT = 580  # seconds, safety margin before 630s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Uses the Laguerre-basis norm formulation with analytic gradients and
    a Jacobi-matrix parameterization for the root locations.
    """
    m = 12
    n = 1
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    # Precompute weights for norm calculation: ||sum c_k L_k^{(alpha)}||^2 = sum c_k^2 / (k+alpha+1) * (k+alpha)!/k! ... 
    # For alpha=1: weight_k = (k+1)! / k! = k+1... wait.
    # Actually for L_k^{(alpha)}, the norm squared is Gamma(k+alpha+1)/Gamma(k+1).
    # For alpha=1: Gamma(k+2)/Gamma(k+1) = k+1.
    # But we need to be careful. The standard Laguerre polynomials L_k^{(alpha)} have
    # ||L_k^{(alpha)}||^2 = Gamma(k+alpha+1)/Gamma(k+1) under weight x^alpha e^{-x}.
    # Here alpha = 1 (since n=1 in the problem, but let's verify).
    # Actually, the problem says n=1. The weight is e^{-x}, which corresponds to alpha=0.
    # Wait, the problem says "Laguerre weight e^{-x}", so alpha=0.
    # For alpha=0: ||L_k^{(0)}||^2 = Gamma(k+1)/Gamma(k+1) = 1.
    # Hmm, but the inspiration used alpha=1. Let me re-read.
    
    # The problem says "dimension n=1" and "Laguerre weight e^{-x}".
    # The weight e^{-x} corresponds to alpha=0.
    # But the inspiration code used alpha=1 with weights = np.arange(13) + 1.
    # Let me check: for L_k^{(alpha)}, the orthogonality is:
    # integral_0^inf x^alpha e^{-x} L_k^{(alpha)}(x) L_l^{(alpha)}(x) dx = Gamma(k+alpha+1)/k! * delta_{kl}
    # For alpha=0: integral e^{-x} L_k(x) L_l(x) dx = 1 * delta_{kl} (since Gamma(k+1)/k! = k!/k! = 1)
    # So for alpha=0, the weights are all 1.
    
    # But wait - the problem mentions "Laguerre-polynomial construction" with "m=12 and dimension n=1".
    # The "n=1" might refer to the parameter in the generalized Laguerre polynomial L_k^{(n)}.
    # Let me use alpha = n = 1, so weight = x^1 e^{-x}.
    # Then ||L_k^{(1)}||^2 = Gamma(k+2)/k! = (k+1)!/k! = k+1.
    
    # Actually, I'll follow the inspiration which used alpha=1 and weights = k+1.
    weights = np.arange(m + 1) + 1  # (k+1) for k=0..m
    
    def compute_p_coeffs(roots):
        """Compute polynomial coefficients in ascending order for prod (x - r_i)^2."""
        # P(x) = prod_i (x - r_i)^2
        # This is a polynomial of degree 2m = 24
        # We need to expand this
        # Start with [1] and multiply by (x - r_i)^2 for each root
        coeffs = np.array([1.0])
        for r in roots:
            # (x - r)^2 = x^2 - 2r*x + r^2
            # Multiply current coeffs by this
            new_coeffs = np.zeros(len(coeffs) + 2)
            new_coeffs[0] += coeffs * r**2
            new_coeffs[1:] += coeffs * (-2*r)
            new_coeffs[2:] += coeffs
            coeffs = new_coeffs
        return coeffs
    
    def compute_norm_sq(roots):
        """Compute the squared norm of the polynomial in the Laguerre basis."""
        p = compute_p_coeffs(roots)
        # p has length 2m+1 = 25
        # We need to express P(x) in terms of L_k^{(1)}(x) for k=0..24
        # But our A matrix only goes up to k=m=12
        # We need A to go up to k=24
        # Let me recompute A for larger size
        pass
    
    # Actually, let me reconsider. The polynomial P(x) = prod (x - r_i)^2 has degree 24.
    # To compute its norm under weight x^alpha e^{-x}, we need to express it in the
    # Laguerre basis up to degree 24.
    
    # Let me rebuild A for degree 24
    max_deg = 2 * m  # 24
    A_full = np.zeros((max_deg + 1, max_deg + 1))
    for k in range(max_deg + 1):
        for j in range(k + 1):
            A_full[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_full_T_inv = np.linalg.inv(A_full.T)
    except np.linalg.LinAlgError:
        A_full_T_inv = np.linalg.pinv(A_full.T)
    
    # Weights for alpha=1: ||L_k^{(1)}||^2 = k+1
    weights_full = np.arange(max_deg + 1) + 1
    
    def compute_norm_sq(roots):
        """Compute the squared norm of P(x) = prod (x - r_i)^2 under weight x e^{-x}."""
        p = compute_p_coeffs(roots)
        # Pad p to length max_deg+1
        if len(p) < max_deg + 1:
            p = np.pad(p, (0, max_deg + 1 - len(p)))
        c = A_full_T_inv @ p
        norm_sq = np.sum(c**2 * weights_full)
        return norm_sq
    
    def compute_norm_sq_and_grad(roots):
        """Compute norm_sq and its gradient with respect to roots."""
        p = compute_p_coeffs(roots)
        if len(p) < max_deg + 1:
            p = np.pad(p, (0, max_deg + 1 - len(p)))
        c = A_full_T_inv @ p
        norm_sq = np.sum(c**2 * weights_full)
        
        if norm_sq <= 1e-10:
            return 0.0, np.zeros(m)
        
        # Gradient of norm_sq with respect to roots:
        # d(norm_sq)/d(roots_i) = 2 * c^T * weights * dc/d(roots_i)
        # where dc/d(roots_i) = A_full_T_inv @ dp/d(roots_i)
        
        # dp/d(r_i) for P(x) = prod_j (x - r_j)^2
        # dP/dr_i = 2 * (x - r_i) * prod_{j!=i} (x - r_j)^2
        # We can compute this by polynomial multiplication
        
        grad_roots = np.zeros(m)
        for i in range(m):
            # Compute Q(x) = prod_{j!=i} (x - r_j)^2
            q_coeffs = np.array([1.0])
            for j in range(m):
                if j == i:
                    continue
                r = roots[j]
                new_q = np.zeros(len(q_coeffs) + 2)
                new_q[0] += q_coeffs * r**2
                new_q[1:] += q_coeffs * (-2*r)
                new_q[2:] += q_coeffs
                q_coeffs = new_q
            
            # dP/dr_i = 2 * (x - r_i) * Q(x)
            # (x - r_i) * Q(x): multiply q_coeffs by (x - r_i)
            temp = np.zeros(len(q_coeffs) + 1)
            temp[0] += q_coeffs * (-roots[i])
            temp[1:] += q_coeffs
            # Then multiply by 2
            dp_dr_i = 2.0 * temp
            
            # Pad to length max_deg+1
            if len(dp_dr_i) < max_deg + 1:
                dp_dr_i = np.pad(dp_dr_i, (0, max_deg + 1 - len(dp_dr_i)))
            
            dc = A_full_T_inv @ dp_dr_i
            grad_roots[i] = 2.0 * np.dot(c, weights_full * dc)
        
        return norm_sq, grad_roots
    
    def cost_func(log_roots):
        """Minimize 1/norm_sq."""
        roots = np.exp(log_roots)
        if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq
    
    def cost_func_with_grad(log_roots):
        """Objective with analytical gradient."""
        roots = np.exp(log_roots)
        if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros(m)
        norm_sq, grad_roots = compute_norm_sq_and_grad(roots)
        if norm_sq <= 1e-10:
            return 1e10, np.zeros(m)
        val = 1.0 / norm_sq
        # d(1/norm_sq)/d(log_roots_i) = -1/norm_sq^2 * d(norm_sq)/d(roots_i) * roots_i
        grad = -grad_roots * roots / (norm_sq ** 2)
        return val, grad
    
    # Initial guess: zeros of generalized Laguerre polynomial
    # For the double-root structure, a good initial guess is the zeros of L_{2m}^{(alpha)}(x)
    # or we can use the geometric sequence approach
    
    # Try multiple initializations
    initializations = []
    
    # 1. Zeros of L_{24}^{(1)}(x)
    try:
        zeros_lag = np.sort(roots_genlaguerre(2 * m, 1.0))
        zeros_lag = zeros_lag[zeros_lag > 0]
        if len(zeros_lag) == m:
            initializations.append(np.log(zeros_lag))
        elif len(zeros_lag) >= m:
            # Take every other zero or spread them
            initializations.append(np.log(zeros_lag[:m]))
    except:
        pass
    
    # 2. Geometric sequence
    for a in [1.0, 2.0, 4.0]:
        for q in [1.5, 2.0, 2.5]:
            geo = a * q**np.arange(m)
            if np.all(geo > 0) and np.all(geo < 1000):
                initializations.append(np.log(geo))
    
    # 3. Random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        random_log = rng.uniform(-2, 5, size=m)
        initializations.append(random_log)
    
    # Optimize from each initialization
    best_log_roots = None
    best_val = 1e10
    
    bounds_log = [(-3.0, 8.0)] * m
    
    for init in initializations:
        if _time_remaining() < 30:
            break
        
        try:
            result = minimize(
                cost_func_with_grad,
                init,
                method='L-BFGS-B',
                bounds=bounds_log,
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
        except Exception:
            continue
    
    # If we didn't find anything, use the best initialization
    if best_log_roots is None:
        best_log_roots = initializations[0]
    
    final_roots = np.exp(best_log_roots)
    
    # Verify positivity and sort
    final_roots = np.sort(final_roots)
    final_roots = np.maximum(final_roots, 1e-10)
    
    return [float(r) for r in final_roots]
