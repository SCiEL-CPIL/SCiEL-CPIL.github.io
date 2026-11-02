import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s limit

    # Factorials: 0! = 1, 4! = 24, 8! = 40320, 12! = 479001600
    fact0 = 1.0
    fact4 = 24.0
    fact8 = 40320.0
    fact12 = 479001600.0

    def compute_integrals(c0, c4, c8):
        """Compute I0, Ix2, Ideriv exactly using Hermite orthogonality."""
        c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
        c = {0: c0, 4: c4, 8: c8, 12: c12}
        degrees = [0, 4, 8, 12]
        
        # I0 = sum_k c_k^2 * k!
        I0 = c0*c0*fact0 + c4*c4*fact4 + c8*c8*fact8 + c12*c12*fact12
        
        # Ix2 = sum over k,l of c_k c_l * <H_k x^2 H_l>
        # x^2 H_k = 0.5*H_{k+2} + k*H_k + 0.5*k*(k-1)*H_{k-2}
        Ix2 = 0.0
        for k in degrees:
            ck = c[k]
            if abs(ck) < 1e-30:
                continue
            # Contribution from 0.5*H_{k+2}
            if (k+2) in c:
                Ix2 += ck * c[k+2] * 0.5 * fact0 if k+2 == 0 else (ck * c[k+2] * 0.5 * (fact4 if k+2==4 else (fact8 if k+2==8 else (fact12 if k+2==12 else 0.0))))
            # Contribution from k*H_k
            Ix2 += ck * c[k] * k * (fact0 if k==0 else (fact4 if k==4 else (fact8 if k==8 else fact12)))
            # Contribution from 0.5*k*(k-1)*H_{k-2}
            if k >= 2 and (k-2) in c:
                Ix2 += ck * c[k-2] * 0.5 * k * (k-1) * (fact0 if k-2==0 else (fact4 if k-2==4 else (fact8 if k-2==8 else 0.0)))
        
        # Ideriv = sum over k,l of c_k c_l * <H_k' H_l'>
        # H_k' = 2k H_{k-1}, so <H_k' H_l'> = 4kl (k-1)! delta_{k,l}
        Ideriv = 0.0
        for k in degrees:
            if k == 0:
                continue
            ck = c[k]
            Ideriv += ck*ck * 4.0 * k * k * (fact0 if k-1==0 else (fact4 if k-1==4 else (fact8 if k-1==8 else 0.0)))
        
        return I0, Ix2, Ideriv

    def objective(coeffs):
        """Compute the uncertainty product A(f)*A(Fhat(f))."""
        c0, c4, c8 = coeffs
        try:
            I0, Ix2, Ideriv = compute_integrals(c0, c4, c8)
            if I0 <= 0 or not np.isfinite(I0):
                return 1e10
            if Ix2 < 0 or not np.isfinite(Ix2):
                return 1e10
            if Ideriv < 0 or not np.isfinite(Ideriv):
                return 1e10
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * math.pi**2)) * (Ideriv / I0)
            
            if A_f_sq <= 0 or A_gf_sq <= 0:
                return 1e10
            
            return math.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return 1e10

    # Initial guess from seed (Goncalves et al. 2017)
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = objective(x0)
    best_coeffs = x0.copy()
    
    # Phase 1: Global Differential Evolution search
    # This is structurally different: population-based, stochastic, global.
    if time.time() < deadline - 30:
        try:
            from scipy.optimize import differential_evolution
            
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
            
            # Use a relatively large population and many iterations for global coverage
            res = differential_evolution(
                objective, 
                bounds=bounds, 
                maxiter=1000, 
                popsize=20, 
                tol=1e-12, 
                mutation=(0.5, 1.0), 
                recombination=0.7, 
                seed=42,
                polish=False  # We will do our own polish
            )
            
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Phase 2: High-precision local polish using Nelder-Mead
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            res = minimize(
                objective, 
                best_coeffs, 
                method='Nelder-Mead', 
                options={'xatol': 1e-15, 'fatol': 1e-17, 'maxiter': 5000, 'adaptive': True}
            )
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final L-BFGS-B polish if time remains
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            res = minimize(
                objective, 
                best_coeffs, 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
