"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from math import factorial, sqrt, pi

# Pre-compute analytic matrices for the objective function
# We want to minimize Delta_x^2 = <x^2> / <1> for f = g * exp(-x^2/2)
# where g = c0 H0 + c4 H4 + c8 H8 + c12 H12
# <1> = sum c_i c_j M_ij
# <x^2> = sum c_i c_j N_ij
# M_ij = int H_i H_j exp(-x^2) dx
# N_ij = int x^2 H_i H_j exp(-x^2) dx

def _hermite_val(n, x):
    """Compute physicists' Hermite polynomial H_n(x) at scalar x."""
    if n == 0:
        return 1.0
    elif n == 1:
        return 2.0 * x
    H_prev = 1.0
    H_curr = 2.0 * x
    for k in range(2, n + 1):
        H_next = 2.0 * x * H_curr - 2.0 * (k - 1) * H_prev
        H_prev = H_curr
        H_curr = H_next
    return H_curr

# Basis indices: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Orthogonality: int H_n H_m exp(-x^2) dx = sqrt(pi) 2^n n! delta_nm
# So M is diagonal.
M_diag = [sqrt(pi) * (2**n) * factorial(n) for n in indices]

# N_ij = int x^2 H_i H_j exp(-x^2) dx
# For our basis {0,4,8,12}, N is diagonal: N_ii = i * M_ii
N_diag = [n * m for n, m in zip(indices, M_diag)]

# c12 is determined by c0, c4, c8 to satisfy g(0)=0
H0_0 = _hermite_val(0, 0)
H4_0 = _hermite_val(4, 0)
H8_0 = _hermite_val(8, 0)
H12_0 = _hermite_val(12, 0)

def _compute_bound_analytic(c0, c4, c8):
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    coeffs = np.array([c0, c4, c8, c12])
    
    # Denominator: sum c_i^2 M_ii
    norm_sq = np.sum(coeffs**2 * np.array(M_diag))
    if norm_sq <= 1e-12:
        return 1e10
        
    # Numerator: sum c_i^2 N_ii
    x2_expectation = np.sum(coeffs**2 * np.array(N_diag))
    
    # Variance Delta_x^2 = <x^2> - <x>^2. <x>=0 due to symmetry.
    # Product Delta_x Delta_xi = Delta_x^2 because Delta_xi = Delta_x.
    return x2_expectation / norm_sq

def _compute_bound_spherical(theta, phi):
    """
    Spherical coordinates:
    c0 = r * sin(theta) * cos(phi)
    c4 = r * sin(theta) * sin(phi)
    c8 = r * cos(theta)
    Since the objective is homogeneous of degree 0 (scale invariant), r cancels.
    We can set r=1.
    theta in [0, pi], phi in [0, 2*pi)
    """
    c0 = np.sin(theta) * np.cos(phi)
    c4 = np.sin(theta) * np.sin(phi)
    c8 = np.cos(theta)
    return _compute_bound_analytic(c0, c4, c8)

def construct():
    """Find optimal coefficients for H0, H4, H8."""
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin before 210s kill
    
    # Initial guess from seed
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_x = x0.copy()
    best_val = _compute_bound_analytic(*x0)
    
    # 1. Try Differential Evolution in spherical coordinates (2D)
    # This removes scale ambiguity and reduces dimensionality
    try:
        bounds_spherical = [(0, np.pi), (0, 2*np.pi)]
        res_de = differential_evolution(
            _compute_bound_spherical,
            bounds=bounds_spherical,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-8,
            polish=True
        )
        if res_de.fun < best_val:
            best_val = res_de.fun
            theta_opt, phi_opt = res_de.x
            best_x = np.array([
                np.sin(theta_opt) * np.cos(phi_opt),
                np.sin(theta_opt) * np.sin(phi_opt),
                np.cos(theta_opt)
            ])
    except Exception:
        pass
    
    # 2. Local refinement with Nelder-Mead in Cartesian coordinates
    if time.time() < deadline - 30:
        try:
            res = minimize(
                lambda x: _compute_bound_analytic(x[0], x[1], x[2]),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # 3. More random restarts if time permits
    rng = np.random.RandomState(42)
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random point on sphere in Cartesian, then project to constraint
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2*np.pi)
        x_start = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        try:
            res = minimize(
                lambda x: _compute_bound_analytic(x[0], x[1], x[2]),
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
    
    return best_x
# EVOLVE-BLOCK-END
