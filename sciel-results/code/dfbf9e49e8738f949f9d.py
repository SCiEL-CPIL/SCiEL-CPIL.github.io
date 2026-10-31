"""
Search for optimal Hermite polynomial coefficients (degrees 0, 4, 8)
to minimize the upper bound on A(f)*A(Fourier(f)) for the uncertainty inequality.

The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
where a12 is chosen so that g(0) = 0.

We use scipy's optimization with analytical gradients to minimize the upper bound.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite, factorial


def _compute_hermite_moments():
    """
    Precompute the matrix C_{k,l} = integral x^2 H_k(x) H_l(x) exp(-x^2) dx
    for k, l in {0, 4, 8, 12}.
    """
    indices = [0, 4, 8, 12]
    n = len(indices)
    C = np.zeros((n, n))
    
    # Formula for C_{k,l}:
    # Non-zero if l == k, l == k+2, or l == k-2.
    # C_{k, k+2} = 0.25 * 2^(k+2) * (k+2)! * sqrt(pi)
    # C_{k, k}   = -(k + 0.5) * 2^k * k! * sqrt(pi)
    # C_{k, k-2} = k*(k-1) * 2^(k-2) * (k-2)! * sqrt(pi)
    
    sqrt_pi = np.sqrt(np.pi)
    
    for i, k in enumerate(indices):
        for j, l in enumerate(indices):
            if l == k + 2:
                C[i, j] = 0.25 * (2 ** (k + 2)) * factorial(k + 2) * sqrt_pi
            elif l == k:
                C[i, j] = -(k + 0.5) * (2 ** k) * factorial(k) * sqrt_pi
            elif l == k - 2:
                C[i, j] = k * (k - 1) * (2 ** (k - 2)) * factorial(k - 2) * sqrt_pi
    
    return C, indices


# Precompute constants
C_MAT, INDICES = _compute_hermite_moments()
# Indices map: 0->0, 1->4, 2->8, 3->12
# We only need to optimize a0, a4, a8. a12 is determined by g(0)=0.
# g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
# H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=4790016
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 4790016.0

# Weights for N = sum a_k^2 * 2^k * k! * sqrt(pi)
# Let W_k = 2^k * k! * sqrt(pi)
W = np.array([
    (2 ** 0) * factorial(0) * np.sqrt(np.pi),
    (2 ** 4) * factorial(4) * np.sqrt(np.pi),
    (2 ** 8) * factorial(8) * np.sqrt(np.pi),
    (2 ** 12) * factorial(12) * np.sqrt(np.pi)
])


# Precompute the matrix D for M_xi: D[i,j] = C[i,j] if i==j, -C[i,j] if |k_i - k_j|==2, else 0
D_MAT = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        if i == j:
            D_MAT[i, j] = C_MAT[i, j]
        elif abs(INDICES[i] - INDICES[j]) == 2:
            D_MAT[i, j] = -C_MAT[i, j]

# Jacobian of a = [a0, a4, a8, a12] with respect to [a0, a4, a8]
# a12 = -(a0 + 12*a4 + 1680*a8) / 4790016
JAC = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0/H12_0, -H4_0/H12_0, -H8_0/H12_0]
])


def _compute_bound_analytic(a0, a4, a8):
    """
    Compute the uncertainty product A(g)*A(hat{g}) analytically.
    """
    # Calculate a12
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    a = np.array([a0, a4, a8, a12])
    
    # N = ||g||^2 = sum a_k^2 * W_k
    N = np.sum(a**2 * W)
    if N < 1e-15:
        return np.inf
    
    # M_x = sum a_k a_l C_{k,l}
    M_x = a @ C_MAT @ a
    
    # M_xi = a^T D a
    M_xi = a @ D_MAT @ a
    
    if M_x < 0 or M_xi < 0:
        return np.inf
        
    return (M_x * M_xi) / (N * N)


def _compute_bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product and its gradient with respect to [a0, a4, a8].
    """
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    a = np.array([a0, a4, a8, a12])
    
    N = np.sum(a**2 * W)
    if N < 1e-15:
        return np.inf, np.array([np.inf, np.inf, np.inf])
    
    M_x = a @ C_MAT @ a
    M_xi = a @ D_MAT @ a
    
    if M_x < 0 or M_xi < 0:
        return np.inf, np.array([np.inf, np.inf, np.inf])
    
    obj = (M_x * M_xi) / (N * N)
    
    # Gradient of obj with respect to a (4-dim)
    # obj = M_x * M_xi / N^2
    # d(obj)/da = (dM_x/da * M_xi * N^2 - M_x * M_xi * 2*N * dN/da) / N^4
    #           = (dM_x/da * M_xi - M_x * M_xi * 2*dN/da / N) / N^2
    #           = M_xi * dM_x/da / N^2 - 2*M_x*M_xi * dN/da / N^3
    
    dN_da = 2 * a * W  # element-wise
    dMx_da = 2 * C_MAT @ a
    dMxi_da = 2 * D_MAT @ a
    
    d_obj_da = (M_xi * dMx_da) / (N * N) - (2.0 * M_x * M_xi * dN_da) / (N * N * N)
    
    # Chain rule: d_obj_dcoeff = JAC.T @ d_obj_da
    grad = JAC.T @ d_obj_da
    
    return obj, grad


def _objective(coeffs):
    """Objective function for optimization."""
    a0, a4, a8 = coeffs
    try:
        val = _compute_bound_analytic(a0, a4, a8)
        if np.isnan(val) or np.isinf(val):
            return np.inf
        return val
    except Exception:
        return np.inf


def _objective_grad(coeffs):
    """Objective function with gradient."""
    a0, a4, a8 = coeffs
    try:
        val, grad = _compute_bound_and_grad(a0, a4, a8)
        if np.isnan(val) or np.isinf(val):
            return np.inf, np.array([np.inf, np.inf, np.inf])
        return val, grad
    except Exception:
        return np.inf, np.array([np.inf, np.inf, np.inf])


def construct():
    """
    Find optimal coefficients a0, a4, a8 to minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    # Start from the seed solution
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_x = x0.copy()
    best_val = _objective(best_x)
    
    # Generate diverse starting points
    np.random.seed(42)
    starts = [x0]
    
    # Random perturbations with varied scales
    for _ in range(80):
        scale = 10 ** np.random.uniform(-4, 1.5)
        x_trial = x0 + np.random.randn(3) * scale
        starts.append(x_trial)
        
    # Structured starts
    starts.append([1.0, -0.1, 0.01])
    starts.append([1.0, 0.1, -0.01])
    starts.append([0.0, 1.0, 0.0])
    starts.append([0.0, 0.0, 1.0])
    starts.append([-1.0, 0.1, -0.01])
    starts.append([0.5, -0.05, 0.005])
    starts.append([-0.5, 0.05, -0.005])
    starts.append([1.0, 0.0, 0.0])
    starts.append([0.0, 1.0, 0.0])
    starts.append([0.0, 0.0, 1.0])
    starts.append([-1.0, 0.0, 0.0])
    starts.append([0.0, -1.0, 0.0])
    starts.append([0.0, 0.0, -1.0])
    
    # First pass: find best starting point
    for start in starts:
        try:
            val = _objective(start)
            if val < best_val:
                best_val = val
                best_x = start.copy()
        except Exception:
            pass
    
    # Second pass: run L-BFGS-B with gradient from each top-5 starting point
    # Sort starts by objective value
    scored_starts = []
    for start in starts:
        try:
            val = _objective(start)
            if np.isfinite(val):
                scored_starts.append((val, start))
        except Exception:
            pass
    
    scored_starts.sort(key=lambda x: x[0])
    top_starts = [s[1] for s in scored_starts[:10]]
    
    for start in top_starts:
        try:
            result = minimize(_objective_grad, start, method='L-BFGS-B',
                             jac=True,
                             options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10,
                                      'maxls': 50})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Refine with tighter tolerances
    try:
        result = minimize(_objective_grad, best_x, method='L-BFGS-B',
                         jac=True,
                         options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12,
                                  'maxls': 100})
        if np.isfinite(result.fun) and result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    # Try Nelder-Mead as a final polish (doesn't need gradient)
    try:
        result = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if np.isfinite(result.fun) and result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
