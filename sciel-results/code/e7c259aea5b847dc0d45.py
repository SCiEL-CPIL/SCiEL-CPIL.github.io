import numpy as np
from scipy.optimize import minimize

def construct():
    # Indices of the Hermite polynomials
    indices = [0, 4, 8, 12]
    
    # Values of H_n(0)
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 168
    # H_12(0) = 10368
    h_vals = np.array([1, 12, 168, 10368])
    
    # Diagonal elements of x^2 and p^2 matrices in this subspace
    # M_xx_ii = i + 0.5
    # M_pp_ii = i + 0.5
    # Since the off-diagonal terms are zero for this specific set of indices (differences are multiples of 4, coupling is 0 or 2)
    diag = np.array([i + 0.5 for i in indices])
    
    def compute_bound(c0, c4, c8):
        # Solve for c12 using g(0) = 0
        # c0*1 + c4*12 + c8*168 + c12*10368 = 0
        c12 = -(c0 + 12*c4 + 168*c8) / 10368.0
        c = np.array([c0, c4, c8, c12])
        
        # Calculate norms
        norm_sq = np.dot(c, c)
        if norm_sq < 1e-12:
            return 1e10 # Avoid division by zero, return large penalty
        
        # Calculate <x^2> and <p^2>
        # Since matrices are diagonal, <x^2> = sum(c_i^2 * diag_i) / sum(c_i^2)
        ex2 = np.dot(c**2, diag) / norm_sq
        ep2 = np.dot(c**2, diag) / norm_sq # Same as ex2
        
        # Product of standard deviations (squared)
        # The task asks for the upper bound on C4, which is likely the product A(f)A(F)
        # Assuming A(f) = sqrt(<x^2>) and A(F) = sqrt(<p^2>)
        # Product = sqrt(ex2 * ep2) = ex2
        # However, the "score" is seed/value.
        # If value is ex2, we minimize ex2.
        return ex2 * ep2 # Minimizing the product of variances is equivalent to minimizing the product of std devs
        
    # Objective function for optimizer
    def objective(x):
        return compute_bound(x[0], x[1], x[2])
    
    # Initial guess from seed
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # We can also try a perturbed guess or a different starting point
    # Based on analysis, increasing c4 helps.
    # Let's run a few optimizations from different starts.
    
    best_val = float('inf')
    best_x = x0
    
    starts = [
        x0,
        np.array([-1.0, 0.1, 0.0]),
        np.array([-1.0, 0.5, 0.0]),
        np.array([-1.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.1]),
        np.array([-1.0, 0.5, 0.1]),
    ]
    
    for s in starts:
        try:
            res = minimize(objective, s, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Return the coefficients c0, c4, c8
    return np.array([best_x[0], best_x[1], best_x[2]])
