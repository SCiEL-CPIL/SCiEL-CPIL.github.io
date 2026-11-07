"""Search for better coefficients for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _compute_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) for given coefficients."""
    try:
        # Hermite polynomials (physicists')
        # H0 = 1
        # H4 = 16x^4 - 48x^2 + 12
        # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
        # H12 = 4096x^12 - 98304x^10 + 921600x^8 - 4032000x^6 + 7257600x^4 - 4608000x^2 + 1330560
        
        # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
        # g(0) = c0*1 + c1*12 + c2*1680 + c3*1330560 = 0
        # c3 = -(c0 + 12*c1 + 1680*c2) / 1330560
        
        c3 = -(c0 + 12.0*c1 + 1680.0*c2) / 1330560.0
        
        # g(x) coefficients from highest to lowest power
        a12 = c3 * 4096.0
        a10 = c3 * (-98304.0)
        a8 = c2 * 256.0 + c3 * 921600.0
        a6 = c2 * (-3072.0) + c3 * (-4032000.0)
        a4 = c1 * 16.0 + c2 * 11520.0 + c3 * 7257600.0
        a2 = c1 * (-48.0) + c2 * (-13440.0) + c3 * (-4608000.0)
        
        # If leading coefficient is negative, flip sign
        if a12 < 0:
            a12 = -a12
            a10 = -a10
            a8 = -a8
            a6 = -a6
            a4 = -a4
            a2 = -a2
        
        # q(x) = g(x)/x^2 = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
        # Find roots of q
        coeffs = np.array([a12, 0.0, a10, 0.0, a8, 0.0, a6, 0.0, a4, 0.0, a2])
        roots = np.roots(coeffs)
        
        # Filter real roots (small imaginary part)
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                real_roots.append(r.real)
        
        if not real_roots:
            return np.inf
        
        # Find largest real root where q changes sign
        best_r = 0.0
        for r in real_roots:
            if r > 1e-10:  # positive root
                # Check if sign changes: evaluate q at r-eps and r+eps
                eps = 1e-8 * max(1.0, abs(r))
                q_minus = (a12*(r-eps)**10 + a10*(r-eps)**8 + a8*(r-eps)**6 + 
                          a6*(r-eps)**4 + a4*(r-eps)**2 + a2)
                q_plus = (a12*(r+eps)**10 + a10*(r+eps)**8 + a8*(r+eps)**6 + 
                         a6*(r+eps)**4 + a4*(r+eps)**2 + a2)
                if q_minus * q_plus < 0:  # sign change
                    if r > best_r:
                        best_r = r
        
        if best_r <= 0:
            return np.inf
        
        return best_r**2 / (2.0 * np.pi)
    except Exception:
        return np.inf


def _value_z(z, c):
    """Evaluate q(x) at x = sqrt(z) where z is the variable, using coefficient array c=[c0,c1,c2]."""
    try:
        c0, c1, c2 = c
        c3 = -(c0 + 12.0*c1 + 1680.0*c2) / 1330560.0
        
        a12 = c3 * 4096.0
        a10 = c3 * (-98304.0)
        a8 = c2 * 256.0 + c3 * 921600.0
        a6 = c2 * (-3072.0) + c3 * (-4032000.0)
        a4 = c1 * 16.0 + c2 * 11520.0 + c3 * 7257600.0
        a2 = c1 * (-48.0) + c2 * (-13440.0) + c3 * (-4608000.0)
        
        if a12 < 0:
            a12 = -a12
            a10 = -a10
            a8 = -a8
            a6 = -a6
            a4 = -a4
            a2 = -a2
        
        # q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
        # At x = sqrt(z): x^2 = z, x^4 = z^2, x^6 = z^3, x^8 = z^4, x^10 = z^5
        return a12*z**5 + a10*z**4 + a8*z**3 + a6*z**2 + a4*z + a2
    except:
        return np.inf


def construct():
    """Search for coefficients that minimize the value using a different approach."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    
    # Seed value
    seed_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed_c.copy()
    best_val = _compute_value(*seed_c)
    
    # New approach: work in z-space where z = x^2
    # q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
    # Let z = x^2: Q(z) = a12*z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
    # We want the largest positive root of Q(z) where Q changes sign
    
    def objective_z(params):
        """Objective in z-space: find coefficients that minimize the largest sign-changing root."""
        try:
            c0, c1, c2 = params
            c3 = -(c0 + 12.0*c1 + 1680.0*c2) / 1330560.0
            
            a12 = c3 * 4096.0
            a10 = c3 * (-98304.0)
            a8 = c2 * 256.0 + c3 * 921600.0
            a6 = c2 * (-3072.0) + c3 * (-4032000.0)
            a4 = c1 * 16.0 + c2 * 11520.0 + c3 * 7257600.0
            a2 = c1 * (-48.0) + c2 * (-13440.0) + c3 * (-4608000.0)
            
            if a12 < 0:
                a12 = -a12
                a10 = -a10
                a8 = -a8
                a6 = -a6
                a4 = -a4
                a2 = -a2
            
            # Find roots of Q(z) = a12*z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
            coeffs = np.array([a12, a10, a8, a6, a4, a2])
            roots = np.roots(coeffs)
            
            # Filter real positive roots
            real_roots = []
            for r in roots:
                if abs(r.imag) < 1e-10 and r.real > 1e-10:
                    real_roots.append(r.real)
            
            if not real_roots:
                return np.inf
            
            # Find largest root where Q changes sign
            best_z = 0.0
            for z in real_roots:
                eps = 1e-8 * max(1.0, abs(z))
                q_minus = (a12*(z-eps)**5 + a10*(z-eps)**4 + a8*(z-eps)**3 + 
                          a6*(z-eps)**2 + a4*(z-eps) + a2)
                q_plus = (a12*(z+eps)**5 + a10*(z+eps)**4 + a8*(z+eps)**3 + 
                         a6*(z+eps)**2 + a4*(z+eps) + a2)
                if q_minus * q_plus < 0:  # sign change
                    if z > best_z:
                        best_z = z
            
            if best_z <= 0:
                return np.inf
            
            # value = r^2 / (2*pi) = z / (2*pi)
            return best_z / (2.0 * np.pi)
        except Exception:
            return np.inf
    
    # Use differential evolution in z-space
    try:
        result = differential_evolution(objective_z, 
                                        bounds=[(-5.0, 5.0), (-1.0, 1.0), (-0.5, 0.5)],
                                        seed=42, maxiter=80, popsize=12, tol=1e-8,
                                        polish=False, init='sobol')
        if result.fun < best_val:
            best_val = result.fun
            best_c = result.x.copy()
    except Exception:
        pass
    
    # Refine with local search
    if time.time() < deadline - 15.0:
        try:
            result = minimize(objective_z, best_c, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Additional random restarts
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        for _ in range(8):
            if time.time() > deadline - 5.0:
                break
            x0 = best_c + rng.normal(0, 0.05, size=3)
            try:
                result = minimize(objective_z, x0, method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-11})
                if result.fun < best_val:
                    best_val = result.fun
                    best_c = result.x.copy()
            except Exception:
                pass
    
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
