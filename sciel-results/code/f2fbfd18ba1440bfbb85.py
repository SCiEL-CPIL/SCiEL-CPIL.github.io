"""Optimized search for Hermite-Gaussian test function coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.special import roots_hermite

def construct():
    """
    Find optimal coefficients (c0, c4, c8) for the Hermite-Gaussian test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is determined by g(0)=0.
    
    Minimizes (sup |x*g(x)|)^2.
    """
    start_time = time.time()
    deadline = 200.0  # Safety margin (210s limit, 10s buffer)

    # Constants for H_n(0)
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Grid setup
    # We need to capture the peak of |x*g(x)|. The Gaussian decays fast.
    # Range [-8, 8] is sufficient.
    N = 131072  # Increased resolution for better peak localization
    x = np.linspace(-8.0, 8.0, N)
    
    # Pre-compute powers
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    
    # Hermite Polynomials (Physicists' convention)
    H0 = np.ones_like(x)
    H4 = 16.0*x4 - 48.0*x2 + 12.0
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
    H12 = (4096.0*x12 - 129024.0*x10 + 1761280.0*x8 - 11942400.0*x6 
           + 37158400.0*x4 - 54432000.0*x2 + 47900160.0)
    
    # Gaussian envelope
    exp_term = np.exp(-x2 / 2.0)
    
    # Basis functions satisfying g(0)=0
    # g(x) = exp(-x^2/2) * [ c0*(H0 - (H0_0/H12_0)H12) + c4*(H4 - (H4_0/H12_0)H12) + c8*(H8 - (H8_0/H12_0)H12) ]
    basis0 = (H0 - (H0_0/H12_0)*H12) * exp_term
    basis4 = (H4 - (H4_0/H12_0)*H12) * exp_term
    basis8 = (H8 - (H8_0/H12_0)*H12) * exp_term
    
    # Vectors for objective: x * g(x)
    xb0 = x * basis0
    xb4 = x * basis4
    xb8 = x * basis8

    def objective(coeffs):
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        max_val = np.max(np.abs(gx))
        return max_val * max_val

    def objective_smooth(coeffs, beta=1000.0):
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        # Soft max approximation: (1/beta) * log(sum(exp(beta * |gx|)))
        # Note: |gx| is non-negative.
        # To avoid overflow, shift by max
        abs_gx = np.abs(gx)
        m = np.max(abs_gx)
        if m == 0:
            return 0.0
        log_sum = m + (1.0/beta) * np.log(np.sum(np.exp(beta * (abs_gx - m))))
        return log_sum * log_sum

    # Initial guess
    x0 = np.array([-1.13, 0.04, 0.0003])
    best_val = objective(x0)
    best_x = x0.copy()

    # 1. Global Search: Differential Evolution
    # Using a tighter box based on the seed and typical scales
    bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]
    
    try:
        if time.time() < deadline - 40:
            de_result = differential_evolution(
                objective, bounds, 
                maxiter=300, 
                popsize=25, 
                seed=42,
                tol=1e-12,
                mutation=(0.3, 1.0),
                crossover=0.8,
                polish=False
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
    except Exception:
        pass

    # 2. Local Refinement: Smooth Objective with BFGS
    # The kink in max(|x|) can hinder gradient-based methods.
    # Using a smooth surrogate (log-sum-exp) allows BFGS to converge to a sharper minimum.
    try:
        if time.time() < deadline - 15:
            # Increase beta for sharper approximation
            beta = 2000.0
            res = minimize(lambda c: objective_smooth(c, beta), best_x, method='BFGS', 
                          options={'maxiter': 500, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_val:
                # Verify with true objective
                true_val = objective(res.x)
                if true_val < best_val:
                    best_val = true_val
                    best_x = res.x.copy()
    except Exception:
        pass

    # 3. Local Refinement: Nelder-Mead on True Objective
    try:
        if time.time() < deadline - 10:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
    except Exception:
        pass

    # 4. Final Polish: Linear Programming Relaxation
    # Minimize M s.t. -M <= x*g(x) <= M
    # This gives the exact sup norm on the grid.
    try:
        if time.time() < deadline - 2:
            # Subsample grid for LP to keep it fast
            # We want to ensure we catch the peak. 
            # If N=131072, step=4 gives ~32k points, which is manageable for LP.
            step = 4
            x_sub = x[::step]
            b0_sub = xb0[::step]
            b4_sub = xb4[::step]
            b8_sub = xb8[::step]
            
            n_vars = 4 # c0, c4, c8, M
            n_constraints = 2 * len(x_sub)
            
            A_ub = np.zeros((n_constraints, n_vars))
            b_ub = np.zeros(n_constraints)
            
            # Constraints: c0*b0 + c4*b4 + c8*b8 - M <= 0
            A_ub[:len(x_sub), 0] = b0_sub
            A_ub[:len(x_sub), 1] = b4_sub
            A_ub[:len(x_sub), 2] = b8_sub
            A_ub[:len(x_sub), 3] = -1.0
            b_ub[:len(x_sub)] = 0.0
            
            # Constraints: -(c0*b0 + c4*b4 + c8*b8) - M <= 0
            A_ub[len(x_sub):, 0] = -b0_sub
            A_ub[len(x_sub):, 1] = -b4_sub
            A_ub[len(x_sub):, 2] = -b8_sub
            A_ub[len(x_sub):, 3] = -1.0
            b_ub[len(x_sub):] = 0.0
            
            # Objective: minimize M
            c_obj = np.array([0.0, 0.0, 0.0, 1.0])
            
            res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
            if res_lp.success:
                # res_lp.x = [c0, c4, c8, M]
                lp_M = res_lp.x[3]
                # Check if this LP solution actually improves the true objective on the full grid
                # The LP minimizes M on the subsample. The true max might be slightly higher on full grid.
                # But if LP M is lower than current sqrt(best_val), it's a strong candidate.
                if lp_M < np.sqrt(best_val):
                    candidate_x = res_lp.x[:3]
                    true_val = objective(candidate_x)
                    if true_val < best_val:
                        best_val = true_val
                        best_x = candidate_x.copy()
                        
    except Exception:
        pass

    # 5. Final High-Precision Nelder-Mead Polish
    try:
        if time.time() < deadline - 0.5:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END
