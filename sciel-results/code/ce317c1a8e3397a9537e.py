import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, basinhopping
from numpy.polynomial import hermite as herm

def _get_hermite_poly_coeffs(n):
    """Get power-basis coefficients of H_n(x) (physicist's Hermite)."""
    # In Hermite basis, H_n is represented as [0]*n + [1]
    h_basis = np.zeros(n + 1)
    h_basis[n] = 1.0
    # Convert to power basis
    power_coeffs = herm.herm2poly(h_basis)
    return power_coeffs

def construct():
    # Precompute Hermite polynomial coefficients
    h0_coeffs = _get_hermite_poly_coeffs(0)  # [1]
    h4_coeffs = _get_hermite_poly_coeffs(4)  # degree 4
    h8_coeffs = _get_hermite_poly_coeffs(8)  # degree 8
    h12_coeffs = _get_hermite_poly_coeffs(12)  # degree 12
    
    # Pad to degree 12
    def pad_to_deg12(coeffs):
        if len(coeffs) < 13:
            return np.concatenate([np.zeros(13 - len(coeffs)), coeffs])
        return coeffs
    
    h0_p = pad_to_deg12(h0_coeffs)
    h4_p = pad_to_deg12(h4_coeffs)
    h8_p = pad_to_deg12(h8_coeffs)
    h12_p = pad_to_deg12(h12_coeffs)
    
    def _max_abs_g(c0, c1, c2):
        c3 = -(c0 + c1 * (-12.0) + c2 * 168.0) / 792.0
        
        # P(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
        # Coefficients from x^12 down to x^0
        p = c0 * h0_p + c1 * h4_p + c2 * h8_p + c3 * h12_p
        
        # P'(x) - x*P(x)
        # P has degree 12, so P' has degree 11, x*P has degree 13
        # P'(x) - x*P(x) has degree 13
        # Since P is even, P' is odd, x*P is odd, so P' - x*P is odd
        # P'(x) - x*P(x) = x * Q(x²) where Q has degree 6
        
        # Let P(x) = sum_{k=0}^{12} p[k] x^k where p[12] is coeff of x^12, etc.
        # Wait, numpy poly1d convention: coeffs[0] is highest degree
        # So p[0] is coeff of x^12, p[1] is coeff of x^11, ..., p[12] is coeff of x^0
        
        # P'(x) = sum_{k=0}^{11} (12-k) * p[k] * x^(11-k)
        # x*P(x) = sum_{k=0}^{12} p[k] * x^(13-k)
        
        # P'(x) - x*P(x):
        # Coeff of x^13: -p[0]
        # Coeff of x^12: 0 (P' has no x^12 term since P is even... wait)
        
        # Actually, since P is even, P(x) = sum_{j=0}^{6} a_j x^(2j)
        # where a_6 = p[0], a_5 = p[2], a_4 = p[4], a_3 = p[6], a_2 = p[8], a_1 = p[10], a_0 = p[12]
        
        # P'(x) = sum_{j=1}^{6} 2j * a_j x^(2j-1)
        # x*P(x) = sum_{j=0}^{6} a_j x^(2j+1)
        
        # P'(x) - x*P(x) = sum_{j=1}^{6} 2j*a_j*x^(2j-1) - sum_{j=0}^{6} a_j*x^(2j+1)
        # = -a_0*x + (2*a_1 - a_0)*x + ... 
        
        # Let me just compute it directly
        # P'(x) coefficients (degree 11 to 0):
        # dP[k] = (12-k) * p[k] for k=0..11, dP[12] = 0
        dP = np.zeros(12)
        for k in range(12):
            dP[k] = (12 - k) * p[k]
        
        # x*P(x) coefficients (degree 13 to 1):
        # xP[k] = p[k] for k=0..12, representing x^(13-k)
        # So xP has 13 coefficients for degrees 13,12,...,1
        
        # P'(x) - x*P(x) has degree 13
        # Coeff of x^13: -p[0]
        # Coeff of x^12: dP[0] - p[1] = 12*p[0] - p[1]
        # Coeff of x^11: dP[1] - p[2] = 11*p[1] - p[2]
        # ...
        # Coeff of x^k: dP[12-k] - p[13-k] for k=1..11
        # Coeff of x^1: dP[11] - p[12] = 1*p[11] - p[12]
        # Coeff of x^0: 0
        
        # Since P is even, p[1], p[3], p[5], p[7], p[9], p[11] are all 0
        # So the polynomial is odd
        
        # Let me construct the full polynomial
        # Degrees 13 down to 0
        full_coeffs = np.zeros(14)
        full_coeffs[0] = -p[0]  # x^13
        for k in range(1, 13):
            # Coeff of x^k
            # From P': if 12-k >= 0 and 12-k <= 11, i.e., k >= 1 and k <= 12
            # dP index: 12-k, value: (12-(12-k))*p[12-k] = k*p[12-k]
            # Wait, I'm getting confused with indexing.
            
            # Let me use a different approach. Just evaluate on a fine grid.
            pass
        
        # Actually, let me just use a hybrid: find critical points by solving the polynomial
        # P'(x) - x*P(x) = 0
        
        # Construct the polynomial explicitly
        # P(x) = p[0]x^12 + p[1]x^11 + ... + p[12]
        # P'(x) = 12*p[0]x^11 + 11*p[1]x^10 + ... + p[11]
        # x*P(x) = p[0]x^13 + p[1]x^12 + ... + p[12]x
        
        # P'(x) - x*P(x) = -p[0]x^13 + (12*p[0] - p[1])x^12 + (11*p[1] - p[2])x^11 + ... + (p[11] - p[12])x
        
        # Coefficients from x^13 to x^0:
        poly_coeffs = np.zeros(14)
        poly_coeffs[0] = -p[0]
        for k in range(1, 13):
            # Coeff of x^k
            # From P': (12-(k-1))*p[k-1] = (13-k)*p[k-1]
            # From x*P: -p[k]
            poly_coeffs[14-k] = (13-k)*p[k-1] - p[k]
        # Coeff of x^0: 0
        
        # Find roots
        roots = np.roots(poly_coeffs)
        # Keep real roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                real_roots.append(r.real)
        
        # Evaluate |g(x)| at each real root
        max_val = 0.0
        for xc in real_roots:
            # Evaluate P(xc)
            px = np.polyval(p, xc)
            g_val = abs(px) * np.exp(-0.5 * xc * xc)
            if g_val > max_val:
                max_val = g_val
        
        return max_val
    
    def _bound(c):
        c0, c1, c2 = c[0], c1 if False else c[1], c[2]
        max_g = _max_abs_g(c0, c1, c2)
        return max_g * max_g
    
    def obj(c):
        try:
            return _bound(c)
        except Exception:
            return 1e12
    
    # Starting points
    starting_points = [
        np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-0.5, 0.04, 0.0]),
        np.array([-1.5, 0.0, 0.001]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-0.8, 0.03, 0.0001]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.3, 0.01, 0.0]),
    ]
    
    best = starting_points[0].copy()
    best_val = obj(best)
    
    for x0 in starting_points:
        try:
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-12, "fatol": 1e-14, "adaptive": True})
            if r.fun < best_val:
                best = r.x.copy()
                best_val = r.fun
        except Exception:
            pass
    
    def local(c):
        try:
            r = minimize(obj, c, method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-13, "fatol": 1e-15, "adaptive": True})
            if r.fun < best_val:
                return (r.fun, r.x, 1)
        except Exception:
            pass
        return (best_val, best, 0)
    
    try:
        bh = basinhopping(obj, best, niter=300,
                          minimizer_kwargs={"method": "Nelder-Mead", "options": {"maxiter": 5000, "xatol": 1e-13, "fatol": 1e-15, "adaptive": True}},
                          stepsize=0.5,
                          accept_test=lambda new, old: new.fun <= old.fun + 1e-10)
        if bh.fun < best_val:
            best = bh.x.copy()
            best_val = bh.fun
    except Exception:
        pass
    
    try:
        r = minimize(obj, best, method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-14, "fatol": 1e-16, "adaptive": True})
        if r.fun < best_val:
            best = r.x.copy()
            best_val = r.fun
    except Exception:
        pass
    
    return np.array(best)
