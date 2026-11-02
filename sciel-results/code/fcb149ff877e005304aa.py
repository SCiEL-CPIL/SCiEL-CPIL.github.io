"""Rewritten program: explicitly optimize the Hermite coefficients to minimize the
uncertainty bound C4 = A(f)*A(Fourier(f)) via stochastic global search (basin hopping)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import differential_evolution


def construct():
    # Grid setup
    # Use a fine grid to accurately capture max and integral
    x = np.linspace(-6.0, 6.0, 100001)
    dx = x[1] - x[0]
    
    # Pre-compute basis functions
    # g(x) = (c0 H0 + c1 H4 + c2 H8 + c3 H12) exp(-x^2/2)
    # c3 = -(c0 H0(0) + c1 H4(0) + c2 H8(0)) / H12(0)
    
    h0 = eval_hermite(0, x)
    h4 = eval_hermite(4, x)
    h8 = eval_hermite(8, x)
    h12 = eval_hermite(12, x)
    
    gauss = np.exp(-0.5 * x * x)
    
    B0 = h0 * gauss
    B1 = h4 * gauss
    B2 = h8 * gauss
    B3 = h12 * gauss
    
    # Values at 0
    H0_0 = eval_hermite(0, 0.0)
    H4_0 = eval_hermite(4, 0.0)
    H8_0 = eval_hermite(8, 0.0)
    H12_0 = eval_hermite(12, 0.0)
    
    # Constraint: c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    # c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0) / H12_0
    K0 = H0_0 / H12_0
    K1 = H4_0 / H12_0
    K2 = H8_0 / H12_0
    
    # Effective basis functions for c0, c1, c2
    # g = c0*(B0 - K0*B3) + c1*(B1 - K1*B3) + c2*(B2 - K2*B3)
    T0 = B0 - K0 * B3
    T1 = B1 - K1 * B3
    T2 = B2 - K2 * B3
    
    # Stack for vectorized operation
    # T is shape (3, N)
    T = np.vstack([T0, T1, T2])
    
    def objective(c):
        # c is shape (3,)
        # g is shape (N,)
        g = np.dot(c, T)
        
        # Calculate L1 norm (integral of absolute value)
        abs_g = np.abs(g)
        L1 = np.sum(abs_g) * dx
        
        if L1 < 1e-12:
            return 1e12
            
        # Calculate Linf norm (max absolute value)
        Linf = np.max(abs_g)
        
        # Objective: (Linf / L1)^2
        # This is scale invariant
        val = (Linf / L1) ** 2
        return val

    # Bounds for coefficients
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    # Use differential evolution for global search
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-8,
            init='sobol'
        )
        best_c = res.x
    except Exception:
        # Fallback to a simple random search or default
        best_c = np.array([-1.0, 0.0, 0.0])
        
    return np.array(best_c)
# EVOLVE-BLOCK-END
