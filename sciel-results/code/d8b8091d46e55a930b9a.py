import numpy as np
from scipy.optimize import minimize
from sympy.polys.orthopolys import hermite_poly
import sympy as sp

# Precompute Hermite polynomials coefficients
def get_hermite_coeffs(n):
    x = sp.Symbol('x')
    Hn = hermite_poly(n, x)
    coeffs = sp.Poly(Hn, x).all_coeffs()
    return [float(c) for c in coeffs]

H0_coeffs = get_hermite_coeffs(0)
H4_coeffs = get_hermite_coeffs(4)
H8_coeffs = get_hermite_coeffs(8)
H12_coeffs = get_hermite_coeffs(12)

# Pad coefficients to degree 12
def pad_coeffs(coeffs, target_degree):
    padded = [0.0] * (target_degree - len(coeffs) + 1)
    padded += coeffs
    return padded

H0_padded = pad_coeffs(H0_coeffs, 12)
H4_padded = pad_coeffs(H4_coeffs, 12)
H8_padded = pad_coeffs(H8_coeffs, 12)
H12_padded = pad_coeffs(H12_coeffs, 12)

def construct():
    c0_seed, c1_seed, c2_seed = -113 / 100, 1 / 25, 1 / 3240
    x0 = np.array([c0_seed, c1_seed, c2_seed])
    
    def evaluate(coeffs):
        c0, c1, c2 = coeffs
        
        # Compute c3 so that g(0) = 0
        val_0 = c0 * H0_padded[-1] + c1 * H4_padded[-1] + c2 * H8_padded[-1]
        c3 = -val_0 / H12_padded[-1]
        
        # Check leading coefficient: g should tend to +infinity at +infinity
        # g(x) ~ c3 * H12_padded[0] * x^12 for large x
        # H12_padded[0] is positive (leading coeff of H12)
        lead_coeff = c3 * H12_padded[0]
        if lead_coeff <= 0:
            # g tends to -infinity, so we'd flip sign. But the problem says "if g tends to 
            # a negative limit at infinity, g is replaced by -g". 
            # However, if we flip g, then the roots of q = g/x^2 change sign behavior.
            # Actually, let's think: if g -> -inf, then -g -> +inf. The evaluator replaces g by -g.
            # So we should compute with -g in that case.
            # Let's just return a large penalty for now and handle the flip properly.
            # Actually, let's handle it: if lead_coeff < 0, use -g
            g_coeffs = c0 * np.array(H0_padded) + c1 * np.array(H4_padded) + c2 * np.array(H8_padded) + c3 * np.array(H12_padded)
            g_coeffs = -g_coeffs  # Flip sign
            # Now leading coeff is positive
            # But c3 effectively becomes -c3 for the purpose of roots
            # The roots of g and -g are the same, but sign change behavior is reversed
            # We need to check: does q change sign at the root?
            # Let's just work with the flipped g
        else:
            g_coeffs = c0 * np.array(H0_padded) + c1 * np.array(H4_padded) + c2 * np.array(H8_padded) + c3 * np.array(H12_padded)
        
        # q = g / x^2. Since g is even (all Hermite polys are even for even n), 
        # g(x) = x^2 * q(x) where q is a polynomial of degree 10.
        # Actually, g has only even powers, so g(x) = sum_{k=0}^{6} a_k x^{2k}
        # g(0) = a_0 = 0, so a_0 = 0.
        # Thus g(x) = x^2 * (a_1 + a_2 x^2 + ... + a_6 x^{10})
        # q(x) = a_1 + a_2 x^2 + ... + a_6 x^{10}
        
        # Extract coefficients of q. g_coeffs is [a_6, a_5, ..., a_1, a_0] for x^12, x^11, ..., x^0
        # Since g is even, odd coefficients are 0.
        # g_coeffs[i] corresponds to x^(12-i)
        # a_k for x^{2k} is at index 12 - 2k
        # a_0 (x^0) at index 12 -> should be 0
        # a_1 (x^2) at index 10
        # a_2 (x^4) at index 8
        # ...
        # a_6 (x^12) at index 0
        
        # q(x) = a_1 + a_2 x^2 + ... + a_6 x^{10}
        # Coefficients of q in standard form (highest to lowest): [a_6, a_5, ..., a_2, a_1]
        q_coeffs = g_coeffs[0:11:2]  # indices 0, 2, 4, 6, 8, 10 -> a_6, a_5, a_4, a_3, a_2, a_1
        
        # Find roots of q
        try:
            q_roots = np.roots(q_coeffs)
        except Exception:
            return None
            
        real_q_roots = q_roots[np.abs(q_roots.imag) < 1e-9].real
        positive_q_roots = real_q_roots[real_q_roots > 1e-10]
        
        if len(positive_q_roots) == 0:
            # No positive roots of q. Check if q(x) > 0 for all x > 0
            # Sample q on a grid
            x_test = np.linspace(0.01, 100, 1000)
            q_vals = np.polyval(q_coeffs, x_test)
            if np.any(q_vals < -1e-9):
                return None  # q goes negative, but no sign change root? That's a problem
            # If q > 0 everywhere, then g > 0 for x > 0, and r doesn't exist in the usual sense
            # The value would be 0? Or we need to interpret differently.
            # Let's assume this gives a very good (small) value
            return 0.0
        
        # For each positive root of q, check if q changes sign there
        # q changes sign at root r if q has odd multiplicity at r
        # We can check by evaluating q on both sides
        
        valid_roots = []
        for r in positive_q_roots:
            # Evaluate q at r - eps and r + eps
            eps = 1e-8 * max(1.0, abs(r))
            q_left = np.polyval(q_coeffs, r - eps)
            q_right = np.polyval(q_coeffs, r + eps)
            if q_left * q_right < 0:
                valid_roots.append(r)
        
        if len(valid_roots) == 0:
            # No sign-changing positive roots
            # Check if g >= 0 for all x >= 0
            x_max = max(50.0, 2.0 * np.max(positive_q_roots) if len(positive_q_roots) > 0 else 50.0)
            x_grid = np.linspace(0, x_max, 5000)
            g_vals = np.polyval(g_coeffs, x_grid)
            if np.any(g_vals < -1e-9):
                return None
            return 0.0
        
        r = max(valid_roots)
        
        # Check non-negativity of g on [0, infinity)
        # g(x) = x^2 * q(x), so g >= 0 iff q >= 0 for x > 0 (and g(0)=0)
        # We need q(x) >= 0 for all x >= 0
        # Sample densely
        x_max = max(50.0, 3.0 * r)
        x_grid = np.linspace(0.001, x_max, 10000)
        q_vals = np.polyval(q_coeffs, x_grid)
        if np.any(q_vals < -1e-8):
            return None
        
        return r ** 2
    
    def objective(coeffs):
        result = evaluate(coeffs)
        if result is None:
            return 1e10
        return result
    
    # Use multiple starting points
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try a few perturbations
    rng = np.random.RandomState(42)
    for i in range(10):
        x_try = x0 + rng.normal(0, 0.01, size=3)
        val_try = objective(x_try)
        if val_try < best_val:
            best_x = x_try
            best_val = val_try
    
    # Optimize from best starting point
    result = minimize(objective, best_x, method='Nelder-Mead', 
                     options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    if result.fun < best_val:
        best_x = result.x
        best_val = result.fun
    
    # Verify the solution
    final_val = objective(best_x)
    if final_val is not None and final_val < 1e10:
        return best_x
    else:
        return x0
