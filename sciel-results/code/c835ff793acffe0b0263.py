import numpy as np
from scipy import optimize
import time
import math

# Precompute constants
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1190.0
H12_0 = 113880.0

norm_sq_0 = math.sqrt(math.pi) * (2**0) * math.factorial(0)
norm_sq_4 = math.sqrt(math.pi) * (2**4) * math.factorial(4)
norm_sq_8 = math.sqrt(math.pi) * (2**8) * math.factorial(8)
norm_sq_12 = math.sqrt(math.pi) * (2**12) * math.factorial(12)

# Precompute grid for L1 norm
# Use a range that covers most of the mass
# exp(-x^2/2) is negligible for |x| > 10
X_GRID = np.linspace(-10, 10, 4000)
DX = X_GRID[1] - X_GRID[0]
GAUSS_HALF = np.exp(-X_GRID**2 / 2.0)

# Precompute Hermite polynomials on the grid
X2 = X_GRID**2
X4 = X2**2
X6 = X4 * X2
X8 = X4**2
X10 = X8 * X2
X12 = X6**2

H0_GRID = np.ones_like(X_GRID)
H4_GRID = 16*X4 - 48*X2 + 12
H8_GRID = 256*X8 - 4480*X6 + 23520*X4 - 30240*X2 + 1190
H12_GRID = 4096*X12 - 129024*X10 + 2031360*X8 - 13040640*X6 + 39916800*X4 - 54432000*X2 + 113880

def _compute_c12(c0, c4, c8):
    return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

def _objective(coeffs):
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    
    # L2 norm squared
    l2_sq = c0**2 * norm_sq_0 + c4**2 * norm_sq_4 + c8**2 * norm_sq_8 + c12**2 * norm_sq_12
    if l2_sq <= 1e-15:
        return 1e10
    l2_norm = np.sqrt(l2_sq)
    
    # L1 norm
    # g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) * exp(-x^2/2)
    # We precomputed H grids and GAUSS_HALF
    g_val = (c0 * H0_GRID + c4 * H4_GRID + c8 * H8_GRID + c12 * H12_GRID) * GAUSS_HALF
    l1_norm = np.trapz(np.abs(g_val), X_GRID)
    
    # Since F(g) = g for this subspace, A(g) = A(F(g))
    # Product = (l1/l2)^2
    return (l1_norm / l2_norm) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 200.0
    
    x0 = np.array([-1.13, 0.04, 0.0003])
    
    best_coeffs = x0.copy()
    best_val = _objective(x0)
    
    def objective_wrapper(coeffs):
        if time.time() > deadline:
            return 1e10
        try:
            return _objective(coeffs)
        except Exception:
            return 1e10
        
    # Optimization
    res = optimize.minimize(objective_wrapper, x0, method='Nelder-Mead', 
                            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
    
    if res.fun < best_val:
        best_val = res.fun
        best_coeffs = res.x
        
    if time.time() < deadline - 10:
        res2 = optimize.minimize(objective_wrapper, best_coeffs, method='L-BFGS-B',
                                 options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        if res2.fun < best_val:
            best_val = res2.fun
            best_coeffs = res2.x

    return np.array(best_coeffs)

if __name__ == "__main__":
    coeffs = construct()
    print(f"Optimal coefficients: {coeffs}")
    print(f"Score: {_objective(coeffs)}")
