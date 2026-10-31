"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, fsolve


def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _compute_matrices():
    """
    Compute moment matrices for the Hermite basis [H0, H4, H8, H12].
    
    M_ij = ∫ H_i H_j e^{-x^2} dx = sqrt(pi) * 2^deg * deg! * delta_ij
    A_ij = ∫ x^2 H_i H_j e^{-x^2} dx (position moment)
    B_ij = ∫ (d/dx H_i)(d/dx H_j) e^{-x^2} dx (momentum/derivative moment)
    """
    degrees = [0, 4, 8, 12]
    n = len(degrees)
    
    M = np.zeros((n, n))
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    
    for i, deg_i in enumerate(degrees):
        M[i, i] = np.sqrt(np.pi) * (2 ** deg_i) * _factorial(deg_i)
        
        # A: using x^2 H_n = (1/4)H_{n+2} + n*H_n + (n/2)H_{n-2}
        for j, deg_j in enumerate(degrees):
            val = 0.0
            if deg_i + 2 == deg_j:
                val += 0.25 * np.sqrt(np.pi) * (2 ** deg_j) * _factorial(deg_j)
            if deg_i == deg_j:
                val += deg_i * np.sqrt(np.pi) * (2 ** deg_i) * _factorial(deg_i)
            if deg_i >= 2 and deg_i - 2 == deg_j:
                val += (deg_i / 2.0) * np.sqrt(np.pi) * (2 ** deg_j) * _factorial(deg_j)
            A[i, j] = val
        
        # B: d/dx H_n = 2n H_{n-1}, so B_ij = 4*i*j * sqrt(pi) * 2^(i-1) * (i-1)! * delta_{i,j} for i,j>0
        for j, deg_j in enumerate(degrees):
            if deg_i > 0 and deg_j > 0 and deg_i == deg_j:
                B[i, j] = 4.0 * deg_i * deg_j * np.sqrt(np.pi) * (2 ** (deg_i - 1)) * _factorial(deg_i - 1)
    
    return M, A, B


def _compute_bound(c0, c4, c8, M, A, B):
    """Compute the uncertainty bound for given coefficients."""
    # c12 is determined by the constraint g(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    c = np.array([c0, c4, c8, c12])
    
    Mc = c @ M @ c
    Ac = c @ A @ c
    Bc = c @ B @ c
    
    if Mc <= 0:
        return 1e10
    return (Ac * Bc) / (Mc * Mc)


def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 210s kill
    
    try:
        M, A, B = _compute_matrices()
        
        # Precompute the linear transformation
        # c = [c0, c4, c8, c12] where c12 = -(c0 + 12*c4 + 1680*c8)/665280
        # With c0 = 1 (scale invariance): c = [1, a, b, -(1+12a+1680b)/665280]
        # c = c_base + a*v_a + b*v_b
        c_base = np.array([1.0, 0.0, 0.0, -1.0 / 665280.0])
        v_a = np.array([0.0, 1.0, 0.0, -12.0 / 665280.0])
        v_b = np.array([0.0, 0.0, 1.0, -1680.0 / 665280.0])
        
        # Precompute quadratic form coefficients
        M0 = c_base @ M @ c_base
        A0 = c_base @ A @ c_base
        B0 = c_base @ B @ c_base
        
        Ma = c_base @ M @ v_a
        Aa = c_base @ A @ v_a
        Ba = c_base @ B @ v_a
        
        Mb = c_base @ M @ v_b
        Ab = c_base @ A @ v_b
        Bb = c_base @ B @ v_b
        
        Maa = v_a @ M @ v_a
        Aaa = v_a @ A @ v_a
        Baa = v_a @ B @ v_a
        
        Mbb = v_b @ M @ v_b
        Abb = v_b @ A @ v_b
        Bbb = v_b @ B @ v_b
        
        Mab = v_a @ M @ v_b
        Aab = v_a @ A @ v_b
        Bab = v_a @ B @ v_b
        
        def obj(a, b):
            Mc = M0 + 2*a*Ma + 2*b*Mb + a*a*Maa + 2*a*b*Mab + b*b*Mbb
            Ac = A0 + 2*a*Aa + 2*b*Ab + a*a*Aaa + 2*a*b*Aab + b*b*Abb
            Bc = B0 + 2*a*Ba + 2*b*Bb + a*a*Baa + 2*a*b*Bab + b*b*Bbb
            if Mc <= 0:
                return 1e10
            return (Ac * Bc) / (Mc * Mc)
        
        # Strategy: Use multiple optimization approaches
        best_score = float('inf')
        best_params = None
        
        # Approach 1: Nelder-Mead from multiple starting points
        start_points = [
            (1.0/25.0, 1.0/3240.0),
            (0.0, 0.0),
            (0.04, 0.0003),
            (0.03, 0.0004),
            (0.05, 0.0002),
            (0.02, 0.0005),
            (0.06, 0.0001),
            (-0.04, 0.0003),
            (0.04, -0.0003),
            (0.01, 0.001),
            (0.1, 0.0001),
            (0.001, 0.01),
        ]
        
        for sp in start_points:
            if time.time() > deadline - 20:
                break
            try:
                result = minimize(lambda x: obj(x[0], x[1]), sp, method='Nelder-Mead',
                                  options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        # Approach 2: L-BFGS-B with gradient (numerical)
        if time.time() < deadline - 30:
            for sp in start_points[:6]:
                try:
                    result = minimize(lambda x: obj(x[0], x[1]), sp, method='L-BFGS-B',
                                      options={'maxiter': 10000, 'gtol': 1e-14})
                    if result.fun < best_score:
                        best_score = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        # Approach 3: BFGS
        if time.time() < deadline - 20:
            for sp in start_points[:4]:
                try:
                    result = minimize(lambda x: obj(x[0], x[1]), sp, method='BFGS',
                                      options={'maxiter': 10000, 'gtol': 1e-14})
                    if result.fun < best_score:
                        best_score = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        # Approach 4: Powell's method
        if time.time() < deadline - 15:
            for sp in start_points[:8]:
                try:
                    result = minimize(lambda x: obj(x[0], x[1]), sp, method='Powell',
                                      options={'maxiter': 10000, 'xtol': 1e-14, 'ftol': 1e-16})
                    if result.fun < best_score:
                        best_score = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        # Approach 5: Differential Evolution for global search
        if time.time() < deadline - 40:
            try:
                bounds = [(-0.5, 0.5), (-0.01, 0.01)]
                result = minimize(lambda x: obj(x[0], x[1]), np.array([0.04, 0.0003]), 
                                  method='differential_evolution', bounds=bounds,
                                  maxiter=200, tol=1e-12, polish=True)
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        # Approach 6: Solve critical point equations directly
        if time.time() < deadline - 10:
            # At optimum: d/da [Ac*Bc/Mc^2] = 0 and d/db [Ac*Bc/Mc^2] = 0
            # This gives two equations in (a, b)
            def critical_equations(x):
                a, b = x
                Mc = M0 + 2*a*Ma + 2*b*Mb + a*a*Maa + 2*a*b*Mab + b*b*Mbb
                Ac = A0 + 2*a*Aa + 2*b*Ab + a*a*Aaa + 2*a*b*Aab + b*b*Abb
                Bc = B0 + 2*a*Ba + 2*b*Bb + a*a*Baa + 2*a*b*Bab + b*b*Bbb
                
                if Mc <= 0:
                    return [1e10, 1e10]
                
                # dMc/da = 2*Ma + 2*a*Maa + 2*b*Mab
                # dAc/da = 2*Aa + 2*a*Aaa + 2*b*Aab
                # dBc/da = 2*Ba + 2*a*Baa + 2*b*Bab
                # dMc/db = 2*Mb + 2*a*Mab + 2*b*Mbb
                # dAc/db = 2*Ab + 2*a*Aab + 2*b*Abb
                # dBc/db = 2*Bb + 2*a*Bab + 2*b*Bbb
                
                dMc_da = 2*Ma + 2*a*Maa + 2*b*Mab
                dAc_da = 2*Aa + 2*a*Aaa + 2*b*Aab
                dBc_da = 2*Ba + 2*a*Baa + 2*b*Bab
                
                dMc_db = 2*Mb + 2*a*Mab + 2*b*Mbb
                dAc_db = 2*Ab + 2*a*Aab + 2*b*Abb
                dBc_db = 2*Bb + 2*a*Bab + 2*b*Bbb
                
                # d(Ac*Bc/Mc^2)/da = [dAc_da * Bc * Mc - Ac * dBc_da * Mc - Ac * Bc * 2 * dMc_da] / Mc^3
                # Setting numerator to zero:
                # dAc_da * Bc - Ac * dBc_da - 2 * Ac * Bc * dMc_da / Mc = 0
                # Multiply by Mc:
                # dAc_da * Bc * Mc - Ac * dBc_da * Mc - 2 * Ac * Bc * dMc_da = 0
                
                eq_a = dAc_da * Bc * Mc - Ac * dBc_da * Mc - 2 * Ac * Bc * dMc_da
                eq_b = dAc_db * Bc * Mc - Ac * dBc_db * Mc - 2 * Ac * Bc * dMc_db
                
                return [eq_a, eq_b]
            
            # Try multiple initial guesses for fsolve
            fsolve_starts = [
                [1.0/25.0, 1.0/3240.0],
                [0.04, 0.0003],
                [0.03, 0.0004],
                [0.05, 0.0002],
                [0.0, 0.0],
            ]
            
            for guess in fsolve_starts:
                if time.time() > deadline - 5:
                    break
                try:
                    sol, info, ier, msg = fsolve(critical_equations, guess, full_output=True)
                    if ier == 1:  # Converged
                        a_sol, b_sol = sol
                        score = obj(a_sol, b_sol)
                        if score < best_score:
                            best_score = score
                            best_params = sol
                except Exception:
                    pass
        
        if best_params is None:
            # Fallback to seed
            return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        
        a_opt, b_opt = best_params
        return np.array([1.0, a_opt, b_opt])
        
    except Exception:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
# EVOLVE-BLOCK-END
