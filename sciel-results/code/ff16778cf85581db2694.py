import numpy as np
from scipy.optimize import minimize
import time
import warnings

warnings.filterwarnings('ignore')

# Precompute grid for fast integration
# The function decays as exp(-x^2/2), so [-10, 10] is sufficient for high precision
X = np.linspace(-12, 12, 20001)
DX = X[1] - X[0]
GAUSS = np.exp(-0.5 * X**2)

# Precompute Hermite polynomials on the grid
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 55440x^4 - 75600x^2 + 10395

X2 = X**2
H0 = np.ones_like(X)
H4 = X2**2 - 6*X2 + 3
H8 = X2**4 - 28*X2**3 + 210*X2**2 - 420*X2 + 105
H12 = X2**6 - 60*X2**5 + 1260*X2**4 - 12600*X2**3 + 55440*X2**2 - 75600*X2 + 10395

def compute_bound_fast(c0, c4, c8):
    """
    Compute the upper bound C4 = A(g) * A(hat_g) efficiently using precomputed grids.
    A(f) = ||f||_1 / ||f||_2
    """
    # c12 is determined by P(0) = 0
    # P(0) = c0*1 + c4*3 + c8*105 + c12*10395 = 0
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
    
    # Construct g(x)
    P_x = c0*H0 + c4*H4 + c8*H8 + c12*H12
    g = P_x * GAUSS
    
    # Construct hat_g(k)
    # Since n is multiple of 4, hat_g(k) = P(k) * exp(-k^2/2)
    # The grid X is symmetric, so we can reuse the precomputed H values
    P_k = c0*H0 + c4*H4 + c8*H8 + c12*H12
    hat_g = P_k * GAUSS
    
    # Compute norms using trapezoidal rule
    # ||g||_1 = sum(|g|) * dx
    # ||g||_2^2 = sum(g^2) * dx
    abs_g = np.abs(g)
    l1_g = np.sum(abs_g) * DX
    l2_g_sq = np.sum(g**2) * DX
    
    abs_hat_g = np.abs(hat_g)
    l1_hat_g = np.sum(abs_hat_g) * DX
    l2_hat_g_sq = np.sum(hat_g**2) * DX
    
    if l2_g_sq < 1e-12 or l2_hat_g_sq < 1e-12:
        return 1e10
        
    # A(g) = l1_g / sqrt(l2_g_sq)
    # A(hat_g) = l1_hat_g / sqrt(l2_hat_g_sq)
    # Product = (l1_g * l1_hat_g) / sqrt(l2_g_sq * l2_hat_g_sq)
    
    return (l1_g * l1_hat_g) / np.sqrt(l2_g_sq * l2_hat_g_sq)

def objective_sphere(theta, phi):
    """
    Objective function on the unit sphere.
    theta in [0, pi], phi in [0, 2*pi]
    """
    c0 = np.sin(theta) * np.cos(phi)
    c4 = np.sin(theta) * np.sin(phi)
    c8 = np.cos(theta)
    return compute_bound_fast(c0, c4, c8)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin for 210s limit
    
    best_score = 1e10
    best_coeffs = np.array([0.0, 0.0, 0.0])
    
    # 1. Coarse Global Search on Sphere
    # Use a Fibonacci lattice for uniform coverage
    N = 2000
    i = np.arange(N)
    # Golden angle
    phi = (1 + np.sqrt(5)) / 2
    # Fibonacci lattice points on sphere
    theta = np.arccos(1 - 2*(i + 0.5)/N)
    phi_angle = 2 * np.pi * i / phi
    
    # Vectorized evaluation for speed
    c0s = np.sin(theta) * np.cos(phi_angle)
    c4s = np.sin(theta) * np.sin(phi_angle)
    c8s = np.cos(theta)
    
    # We can't easily vectorize compute_bound_fast as is without significant memory overhead for 2000 points
    # But we can loop over a subset or just do a simpler grid
    # Let's do a simpler grid first to find a good region
    n_grid = 50
    theta_grid = np.linspace(0, np.pi, n_grid)
    phi_grid = np.linspace(0, 2*np.pi, n_grid, endpoint=False)
    
    local_best = 1e10
    local_best_params = (0, 0)
    
    for t in theta_grid:
        for p in phi_grid:
            score = objective_sphere(t, p)
            if score < local_best:
                local_best = score
                local_best_params = (t, p)
                if score < best_score:
                    best_score = score
                    best_coeffs = np.array([
                        np.sin(t)*np.cos(p), 
                        np.sin(t)*np.sin(p), 
                        np.cos(t)
                    ])
    
    # 2. Local Refinement
    # Use L-BFGS-B or Nelder-Mead on the sphere parameters
    if time.time() < deadline - 60:
        try:
            res = minimize(objective_sphere, local_best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                t, p = res.x
                best_coeffs = np.array([
                    np.sin(t)*np.cos(p), 
                    np.sin(t)*np.sin(p), 
                    np.cos(t)
                ])
        except Exception:
            pass

    # 3. Multi-start Local Optimization in Cartesian Space
    # The sphere parameterization can be tricky near poles. 
    # Let's also try optimizing the raw coefficients directly from several random starts.
    # We normalize the result to unit length to maintain scale invariance.
    
    def objective_cart(c):
        norm = np.linalg.norm(c)
        if norm < 1e-10:
            return 1e10
        c_norm = c / norm
        return compute_bound_fast(c_norm[0], c_norm[1], c_norm[2])
    
    # Random restarts
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random direction on sphere
        u = np.random.randn(3)
        u = u / np.linalg.norm(u)
        
        try:
            res = minimize(objective_cart, u, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                # Normalize the best coefficients found
                best_coeffs = res.x / np.linalg.norm(res.x)
        except Exception:
            pass

    # 4. Final Polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective_cart, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x / np.linalg.norm(res.x)
        except Exception:
            pass

    # Return the three coefficients c0, c4, c8
    # The evaluator expects the coefficients for H0, H4, H8.
    # The scale doesn't matter for the score (ratio of norms), but we return the normalized version.
    return best_coeffs

# EVOLVE-BLOCK-END
