import numpy as np
from scipy.optimize import differential_evolution, minimize

def get_P_coeffs(c1, c2):
    """
    Returns coefficients of P(y) = q(sqrt(y)) in ascending order [F, E, D, C, B, A]
    for c0 = 1.
    g(x) = H0 + c1*H4 + c2*H8 + c3*H12
    c3 = -(1 + 12*c1 + 1680*c2) / 10321920
    q(x) = g(x)/x^2
    P(y) = q(sqrt(y))
    """
    c0 = 1.0
    c3 = -(c0 + 12.0*c1 + 1680.0*c2) / 10321920.0
    
    # Hermite polys in x
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12 = 4096x^12 - 98304x^10 + 1032192x^8 - 4915200x^6 + 8709120x^4 - 5529600x^2 + 10321920
    
    # g(x) coefficients in x^2, x^4, ..., x^12
    # Let y = x^2
    # H0 = 1
    # H4 = 16y^2 - 48y + 12
    # H8 = 256y^4 - 3072y^3 + 11520y^2 - 13440y + 1680
    # H12 = 4096y^6 - 98304y^5 + 1032192y^4 - 4915200y^3 + 8709120y^2 - 5529600y + 10321920
    
    # g(y) = 1*H0 + c1*H4 + c2*H8 + c3*H12
    # g(y) is a polynomial in y of degree 6.
    # g(y) = A6*y^6 + A5*y^5 + A4*y^4 + A3*y^3 + A2*y^2 + A1*y + A0
    # Since g(0)=0, A0=0.
    # q(x) = g(x)/x^2 = g(y)/y = A6*y^5 + A5*y^4 + A4*y^3 + A3*y^2 + A2*y + A1
    # So P(y) = A6*y^5 + ... + A1
    
    # Coefficients of Hk in y
    H0_y = np.array([1.0]) # degree 0
    H4_y = np.array([12.0, -48.0, 16.0]) # degree 2
    H8_y = np.array([1680.0, -13440.0, 11520.0, -3072.0, 256.0]) # degree 4
    H12_y = np.array([10321920.0, -5529600.0, 8709120.0, -4915200.0, 1032192.0, -98304.0, 4096.0]) # degree 6
    
    # Pad to degree 6
    H0_y = np.pad(H0_y, (0, 6), 'constant')
    H4_y = np.pad(H4_y, (0, 4), 'constant')
    H8_y = np.pad(H8_y, (0, 2), 'constant')
    
    g_y = c0*H0_y + c1*H4_y + c2*H8_y + c3*H12_y
    
    # g_y[0] should be ~0
    # P(y) = g_y[1:] / 1.0  (since g_y[0] is constant term which is 0, we divide by y)
    # Wait, g_y is coeffs of g(y) in ascending order.
    # g(y) = g_y[0] + g_y[1]y + ... + g_y[6]y^6
    # q(x) = g(x)/x^2 = (g_y[0] + g_y[1]y + ...)/y = g_y[1] + g_y[2]y + ... + g_y[6]y^5
    # So P(y) coeffs are g_y[1:]
    
    P_coeffs = g_y[1:]
    return P_coeffs

def objective(c):
    c1, c2 = c
    P_coeffs = get_P_coeffs(c1, c2)
    
    # Find roots of P(y)
    # P_coeffs are in ascending order. np.roots expects descending order.
    P_desc = P_coeffs[::-1]
    
    # Handle case where leading coeff is 0
    # Trim leading zeros
    while len(P_desc) > 1 and abs(P_desc[0]) < 1e-12:
        P_desc = P_desc[1:]
        
    if len(P_desc) == 1:
        # Constant polynomial, no roots
        return 0.0
        
    roots = np.roots(P_desc)
    
    max_y = 0.0
    for r in roots:
        if abs(r.imag) < 1e-8:
            y = r.real
            if y > 1e-8: # Only positive roots
                # Check sign change
                # Evaluate P(y-delta) and P(y+delta)
                delta = 1e-6 * (1 + abs(y))
                # Use Horner or np.polyval
                # P_desc is descending
                val_minus = np.polyval(P_desc, y - delta)
                val_plus = np.polyval(P_desc, y + delta)
                
                if val_minus * val_plus < 0:
                    if y > max_y:
                        max_y = y
                        
    return max_y / (2.0 * np.pi)

def construct():
    # Scale invariance allows fixing c0=1
    # Optimize c1, c2
    # Seed: c0=-1.13, c1=0.04, c2=1/3240
    # Normalized seed: c1_norm = 0.04 / -1.13, c2_norm = (1/3240) / -1.13
    # Let's just use the normalized values as starting point for DE
    
    c0_seed = -1.13
    c1_seed = 0.04
    c2_seed = 1.0/3240.0
    
    c1_start = c1_seed / c0_seed
    c2_start = c2_seed / c0_seed
    
    bounds = [(-10.0, 10.0), (-1.0, 1.0)]
    
    # Differential Evolution
    result = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-8, polish=True, popsize=20)
    
    best_c1, best_c2 = result.x
    best_val = result.fun
    
    # Try to refine with Nelder-Mead or BFGS
    # If best_val is 0, we are done.
    if best_val > 1e-10:
        try:
            res_local = minimize(objective, [best_c1, best_c2], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
            if res_local.fun < best_val:
                best_c1, best_c2 = res_local.x
                best_val = res_local.fun
        except:
            pass
            
    # Return [c0, c1, c2]
    # We can return any valid scaling. c0=1 is fine.
    return [1.0, float(best_c1), float(best_c2)]
