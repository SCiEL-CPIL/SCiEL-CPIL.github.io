# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute the Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

# Coefficients for H_n(x) in terms of x^2, i.e., H_n(x) = sum a_k (x^2)^k
# H0(x) = 1 => [1]
# H4(x) = 12 - 48x^2 + 16x^4 => [12, -48, 16]
# H8(x) = 168 - 1344x^2 + 2688x^4 - 1344x^6 + 256x^8 => [168, -1344, 2688, -1344, 256]
# H12(x) = 18480 - 921600x^2 + 1424640x^4 - 1003520x^6 + 332160x^8 - 46080x^10 + 4096x^12
# => [18480, -921600, 1424640, -1003520, 332160, -46080, 4096]

H0_c = np.array([1.0])
H4_c = np.array([12.0, -48.0, 16.0])
H8_c = np.array([168.0, -1344.0, 2688.0, -1344.0, 256.0])
H12_c = np.array([18480.0, -921600.0, 1424640.0, -1003520.0, 332160.0, -46080.0, 4096.0])

H12_0 = H12_c[0]

def get_q_coeffs(c0, c1, c2):
    """
    Returns coefficients of q(y) = g(x)/x^2 where y=x^2.
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    c3 is fixed such that g(0)=0.
    q(y) is a polynomial in y of degree 5.
    """
    c3 = -(c0 * H0_c[0] + c1 * H4_c[0] + c2 * H8_c[0]) / H12_0
    
    # g(x) coefficients for x^0, x^2, ..., x^12
    # We only need x^2 to x^12 for q(y)
    # g(x) = sum_{k=0}^6 g_k (x^2)^k
    # q(y) = sum_{k=1}^6 g_k y^{k-1} = sum_{j=0}^5 g_{j+1} y^j
    
    # Build g coefficients
    # H0 contributes to k=0
    # H4 contributes to k=0,1,2
    # H8 contributes to k=0..4
    # H12 contributes to k=0..6
    
    g_k = np.zeros(7)
    
    # c0 * H0
    g_k[0] += c0 * H0_c[0]
    
    # c1 * H4
    for i, coeff in enumerate(H4_c):
        g_k[i] += c1 * coeff
        
    # c2 * H8
    for i, coeff in enumerate(H8_c):
        g_k[i] += c2 * coeff
        
    # c3 * H12
    for i, coeff in enumerate(H12_c):
        g_k[i] += c3 * coeff
        
    # q(y) coefficients are g_1 to g_6
    q_coeffs = g_k[1:7]
    return q_coeffs

def evaluate_value(c0, c1, c2):
    """
    Compute the value = r^2 / (2*pi) where r is the largest real root of q
    at which q changes sign.
    Returns 0.0 if q(y) >= 0 for all y >= 0.
    """
    q_coeffs = get_q_coeffs(c0, c1, c2)
    
    # Find roots of q(y) = 0
    # q_coeffs are [q_0, ..., q_5] for y^0 to y^5
    # np.roots expects [q_5, ..., q_0]
    poly_coeffs = np.flip(q_coeffs)
    
    try:
        roots = np.roots(poly_coeffs)
    except Exception:
        return float('inf')
    
    valid_roots = []
    for r in roots:
        if np.isreal(r):
            r_real = np.real(r)
            if r_real >= 0:
                valid_roots.append(r_real)
                
    if not valid_roots:
        # No positive real roots. Check if q(y) >= 0 for all y >= 0
        # Sample a few points
        y_test = np.linspace(0, 100, 100)
        q_vals = np.polyval(poly_coeffs, y_test)
        if np.all(q_vals >= -1e-9):
            return 0.0
        # If it dips below, there must be a root we missed or it's negative everywhere?
        # If no roots and negative, it's always negative. 
        # But q(0) = c0*0 + ... wait.
        # If q(y) < 0 for all y, it doesn't change sign? 
        # The problem says "largest real root of q at which q changes sign".
        # If no sign change, value is 0? Or inf?
        # "If g tends to a negative limit at infinity, g is replaced by -g."
        # This ensures g(x) -> +inf as x->inf. So q(y) -> +inf as y->inf.
        # If q(y) has no real roots and q(inf)>0, then q(y) > 0 for all y.
        # So value should be 0.
        return 0.0

    # Check sign change for each valid root
    def q_eval(y):
        return np.polyval(poly_coeffs, y)
    
    sign_change_roots = []
    for r in valid_roots:
        if r < 1e-9:
            val_after = q_eval(1e-8)
            val_before = q_eval(0) # q(0)
            # Note: q(0) is g_1.
            if val_before * val_after < 0:
                sign_change_roots.append(r)
        else:
            eps = 1e-7 * max(1.0, r)
            val_before = q_eval(r - eps)
            val_after = q_eval(r + eps)
            if val_before * val_after < 0:
                sign_change_roots.append(r)
                
    if not sign_change_roots:
        return 0.0
        
    max_y = max(sign_change_roots)
    value = max_y / (2 * np.pi)
    return value

def construct():
    """
    Structural Strategy:
    1. The optimal solution for minimizing the largest sign-changing root often lies on the boundary of the region where the polynomial is non-negative.
    2. The boundary is characterized by a double root at some y* > 0.
    3. We parametrize the search by y*. For each y*, we solve for c0, c1, c2 such that q(y*)=0 and q'(y*)=0.
    4. We check if the resulting polynomial is non-negative for all y >= 0.
    5. If yes, value is 0.
    6. If not, we keep the best value found.
    """
    start_time = time.time()
    deadline = start_time + 180.0
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_value = evaluate_value(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    
    # If seed is already 0, return it
    if best_value < 1e-9:
        return best_coeffs
    
    # Define the linear system for c0, c1, c2 given a double root at y*
    # q(y) = c0 * Q0(y) + c1 * Q1(y) + c2 * Q2(y)
    # where Q_i(y) are the q-coefficients for H_i.
    # We need q(y*) = 0 and q'(y*) = 0.
    # This is a 2x3 system. We fix c2 = 1 and solve for c0, c1.
    
    def solve_for_c(y_star):
        """
        Solves for c0, c1 given c2=1 such that q(y*)=0 and q'(y*)=0.
        Returns (c0, c1, 1.0) or None if no solution.
        """
        # Get basis polynomials for q
        # q_c0(y) = get_q_coeffs(1, 0, 0)
        # q_c1(y) = get_q_coeffs(0, 1, 0)
        # q_c2(y) = get_q_coeffs(0, 0, 1)
        
        q0 = get_q_coeffs(1.0, 0.0, 0.0)
        q1 = get_q_coeffs(0.0, 1.0, 0.0)
        q2 = get_q_coeffs(0.0, 0.0, 1.0)
        
        # Evaluate q and q' at y_star
        # q(y) = sum q_k y^k
        # q'(y) = sum k q_k y^{k-1}
        
        def eval_q(qc, y):
            # Horner's
            res = 0.0
            for coeff in reversed(qc):
                res = res * y + coeff
            return res
        
        def eval_q_deriv(qc, y):
            # Derivative coefficients: d_k = (k+1) q_{k+1} for power k
            # Or just compute analytically
            res = 0.0
            for k in range(1, len(qc)):
                res += k * qc[k] * (y ** (k-1))
            return res
        
        # System:
        # c0 * q0(y*) + c1 * q1(y*) + 1 * q2(y*) = 0
        # c0 * q0'(y*) + c1 * q1'(y*) + 1 * q2'(y*) = 0
        
        a00 = eval_q(q0, y_star)
        a01 = eval_q(q1, y_star)
        b0 = -eval_q(q2, y_star)
        
        a10 = eval_q_deriv(q0, y_star)
        a11 = eval_q_deriv(q1, y_star)
        b1 = -eval_q_deriv(q2, y_star)
        
        det = a00 * a11 - a01 * a10
        if abs(det) < 1e-12:
            return None
            
        c0 = (b0 * a11 - a01 * b1) / det
        c1 = (a00 * b1 - b0 * a10) / det
        
        return np.array([c0, c1, 1.0])

    # Scan y_star
    # y* = r^2. r is the root.
    # Likely range for y*: 0 to 100?
    # Use a logarithmic and linear grid.
    
    y_grid = np.concatenate([
        np.linspace(0.01, 10, 500),
        np.logspace(-2, 2, 200)
    ])
    y_grid = np.unique(y_grid)
    
    for y_star in y_grid:
        if time.time() > deadline - 10.0:
            break
            
        c = solve_for_c(y_star)
        if c is None:
            continue
            
        # Check validity: q(y) >= 0 for all y >= 0
        # We can check by finding all real roots of q(y) and ensuring no sign changes 
        # or that all roots are double roots (even multiplicity).
        
        val = evaluate_value(c[0], c[1], c[2])
        
        if val < best_value:
            best_value = val
            best_coeffs = c
            
        if best_value < 1e-9:
            return best_coeffs

    # If no non-negative polynomial found via double root scan, 
    # try a few random points or the seed if it's better.
    # But the double root scan should cover the boundary.
    
    return best_coeffs
# EVOLVE-BLOCK-END
