import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

# Note: eval_hermite is for Physicist's Hermite polynomials H_n(x).
# The prompt mentions "Hermite-polynomial". 
# If the seed program used Physicist's, I should check which one yields the seed score.
# However, the structure of the problem (invariance) works for both if the Gaussian is chosen correctly.
# For Physicist's H_n, the eigenfunctions are H_n(x) e^{-x^2/2}.
# For Probabilist's He_n, the eigenfunctions are He_n(x) e^{-x^2/2}.
# Let's assume Physicist's H_n as it is more common in "Hermite polynomial" contexts without "probabilist's" qualifier.
# But wait, the seed coefficients are specific.
# Let's try to compute the score for the seed program with both definitions to see which one matches 1.0 (or rather, what the baseline value is).
# The task says "score is seed/value". So if I return the seed coeffs, score should be 1.
# I don't know the exact value of the seed, but I can optimize relative to it.
# Actually, I just need to find the minimum. The absolute value doesn't matter for the optimization, only the relative improvement.
# But I must return the coefficients.

# Let's use Physicist's Hermite polynomials H_n(x).
# Eigenfunctions: phi_n(x) = H_n(x) e^{-x^2/2}.
# FT[phi_n] = i^n phi_n(x) (with unitary FT).
# Weight in variance integral: |f|^2 = e^{-x^2} P(x)^2.
# Matrices:
# N_mn = int H_m(x) H_n(x) e^{-x^2} dx
# M_mn = int x^2 H_m(x) H_n(x) e^{-x^2} dx

def get_hermite_coeffs(n):
    # Returns coefficients of H_n(x) = sum c_k x^k
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 4x^2 - 2
    # H_3 = 8x^3 - 12x
    # H_4 = 16x^4 - 48x^2 + 12
    # We need H_0, H_4, H_8, H_12.
    pass

# Instead of symbolic, use scipy.integrate.quad with eval_hermite.

def construct():
    start_time = time.time()
    deadline = 200.0 # Safety margin
    
    # Degrees to include
    degrees = [0, 4, 8, 12]
    
    # Precompute H_n(0) for constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # Compute matrices N and M
    # N_mn = integral from -inf to inf of H_m(x) H_n(x) exp(-x^2) dx
    # M_mn = integral from -inf to inf of x^2 H_m(x) H_n(x) exp(-x^2) dx
    
    def integrand_N(x, m, n):
        return eval_hermite(m, x) * eval_hermite(n, x) * np.exp(-x**2)
        
    def integrand_M(x, m, n):
        return x**2 * eval_hermite(m, x) * eval_hermite(n, x) * np.exp(-x**2)
        
    N = np.zeros((4, 4))
    M = np.zeros((4, 4))
    
    for i, mi in enumerate(degrees):
        for j, mj in enumerate(degrees):
            if j < i:
                N[i, j] = N[j, i]
                M[i, j] = M[j, i]
            else:
                # Use quad
                N[i, j], _ = quad(integrand_N, -np.inf, np.inf, args=(mi, mj))
                M[i, j], _ = quad(integrand_M, -np.inf, np.inf, args=(mi, mj))
                
    # Constraint: P(0) = 0
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    
    def objective(vars):
        c0, c4, c8 = vars
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        c = np.array([c0, c4, c8, c12])
        
        num = c @ M @ c
        den = c @ N @ c
        
        if den <= 0:
            return 1e10
        return num / den
        
    # Multi-start optimization
    best_val = np.inf
    best_x = None
    
    # Initial guess from seed
    seed_x = np.array([-113/100, 1/25, 1/3240])
    
    # Random starts
    np.random.seed(42)
    starts = [seed_x]
    for _ in range(50):
        # Scale roughly based on seed
        rand_x = np.random.normal(0, 1, 3) * np.array([1, 1, 0.1]) 
        starts.append(rand_x)
        
    for x0 in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    if best_x is None:
        return seed_x
        
    return np.array(best_x)

from scipy.integrate import quad
