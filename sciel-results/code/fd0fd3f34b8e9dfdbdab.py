# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).
    
    Strategy:
    1. Exploit symmetry: set x_i = y_i. Then V = integral_0^1 Ux(c)^2 dc.
    2. The 1D union length U(c) is a piecewise-linear function of c with 
       breakpoints at values where the ordering of interval endpoints changes.
    3. Compute the exact integral of U(c)^2 by finding all breakpoints and 
       integrating piecewise.
    4. Use gradient-free optimization (Nelder-Mead) with a fast, exact objective.
    """
    N = 8
    deadline = time.time() + 580.0

    i_vals = np.arange(N, dtype=np.float64)
    shear = i_vals / N  # shape (N,)
    width = 1.0 / N

    def exact_objective(x):
        """
        Compute integral_0^1 U(c)^2 dc exactly, where U(c) is the 1D union length 
        of intervals [x_i + c * shear_i, x_i + c * shear_i + width].
        """
        # Build arrays of coefficients: E_p(c) = a_p * c + b_p
        # For left endpoint i: a = shear_i, b = x_i
        # For right endpoint i: a = shear_i, b = x_i + width
        
        a_coeff = np.concatenate([shear, shear])  # shape (2N,)
        b_coeff = np.concatenate([x, x + width])   # shape (2N,)
        
        # Find all c where any two endpoints are equal
        breakpoints = set()
        breakpoints.add(0.0)
        breakpoints.add(1.0)
        
        for p in range(2 * N):
            for q in range(p + 1, 2 * N):
                da = a_coeff[p] - a_coeff[q]
                db = b_coeff[p] - b_coeff[q]
                if abs(da) < 1e-15:
                    continue
                c_val = -db / da
                if 0.0 <= c_val <= 1.0:
                    breakpoints.add(c_val)
        
        breakpoints = np.sort(np.array(list(breakpoints), dtype=np.float64))
        
        if len(breakpoints) < 2:
            return 0.0
        
        # Integrate U(c)^2 over each subinterval [breakpoints[k], breakpoints[k+1]]
        total_integral = 0.0
        for idx in range(len(breakpoints) - 1):
            c0 = breakpoints[idx]
            c1 = breakpoints[idx + 1]
            
            if c1 - c0 < 1e-15:
                continue
            
            # Compute U(c) at c0 and c1
            def union_len(c):
                L = x + c * shear
                R = L + width
                order = np.argsort(L)
                L_s = L[order]
                R_s = R[order]
                max_r = -np.inf
                u_len = 0.0
                for ii in range(N):
                    if L_s[ii] > max_r:
                        u_len += R_s[ii] - L_s[ii]
                        max_r = R_s[ii]
                    else:
                        max_r = max(max_r, R_s[ii])
                return u_len
            
            u0 = union_len(c0)
            u1 = union_len(c1)
            
            m = (u1 - u0) / (c1 - c0)
            b = u0 - m * c0
            
            integral = (m ** 2) * (c1 ** 3 - c0 ** 3) / 3.0 + \
                       (2 * m * b) * (c1 ** 2 - c0 ** 2) / 2.0 + \
                       (b ** 2) * (c1 - c0)
            
            total_integral += integral
        
        return total_integral

    # Initial guess: centered symmetric configuration
    k = np.arange(N, dtype=np.float64)
    x0 = -k / (2.0 * N)

    # Optimize using Nelder-Mead
    best_x = x0.copy()
    best_val = exact_objective(x0)

    # Try multiple initializations (wider spread to escape local minima)
    inits = [
        x0.copy(),
        -k / (2.0 * N) * 0.5,
        -k / (2.0 * N) * 1.5,
        np.linspace(0, 1, N) - 0.5,
        np.linspace(-0.5, 0.5, N),
        np.linspace(-0.25, 0.25, N),
        np.linspace(-0.75, 0.75, N),
        -k / (2.0 * N) * 0.75,
        -k / (2.0 * N) * 1.25,
    ]

    for x_start in inits:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(exact_objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-13,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # High-precision polish: several rounds with progressively tighter tolerances
    polish_iters = [
        (8000, 1e-12, 1e-14),
        (12000, 1e-13, 1e-15),
        (15000, 1e-14, 1e-16),
    ]
    for maxit, xatol, fatol in polish_iters:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(exact_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': maxit, 'xatol': xatol, 'fatol': fatol,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Construct the 64x2 array
    positions = np.zeros((64, 2), dtype=np.float64)
    for k_idx in range(64):
        i = k_idx // 8
        j = k_idx % 8
        positions[k_idx, 0] = best_x[i]
        positions[k_idx, 1] = best_x[j]

    return positions

# EVOLVE-BLOCK-END
