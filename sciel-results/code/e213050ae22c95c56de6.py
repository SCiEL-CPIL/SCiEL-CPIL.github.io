import numpy as np
import time
import math
from scipy.special import factorial
from scipy.optimize import minimize, differential_evolution
import sys

def rudin_problem10() -> np.ndarray:
    """
    Finds the coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimizes the ratio R = ||f||_4 / ||f||_2.
    
    Strategy:
    1. Precompute quadrature nodes and weights for exact integration of degree 20 polynomials
       (since |f|^4 is degree 40? No, f is degree 10, f^2 is degree 20, f^4 is degree 40.
       Wait, the prompt says "exact for this degree". Let's check.
       f is a linear combination of degree 10 harmonics.
       f^2 is a combination of degrees 0, 2, ..., 20.
       f^4 is a combination of degrees 0, 4, ..., 40.
       Gauss-Legendre with 70 points is exact for polynomials of degree 139 in cos(theta).
       The phi integration is exact for any degree in phi if we use enough points (130 is plenty for m=10).
       So the quadrature is exact.
    2. Use Differential Evolution for global exploration.
    3. Use BFGS for local polishing.
    4. Time-guarded to return best result before 630s.
    """
    
    k = 10
    n_coeffs = 2 * k + 1
    
    # --- Precomputation ---
    
    # 1. Gauss-Legendre nodes and weights for cos(theta)
    # 70 points are sufficient for exactness of degree 40 in cos(theta) (needs > 20 points)
    # Actually, standard GL(n) is exact for degree 2n-1. n=70 -> exact for degree 139.
    # The highest degree in cos(theta) for f^4 is 40. So 70 is more than enough.
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl is in [-1, 1], which is cos(theta)
    # We need to convert to theta? No, the measure is sin(theta) dtheta dphi.
    # Let u = cos(theta). du = -sin(theta) dtheta.
    # Integral over sphere = Integral_{-1}^{1} Integral_{0}^{2pi} f(theta, phi) sin(theta) dphi dtheta
    # = Integral_{-1}^{1} Integral_{0}^{2pi} f(arccos(u), phi) dphi du
    # So we can integrate directly over u = cos(theta) with standard GL weights.
    
    # 2. Equally spaced phi points
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # 3. Spherical Harmonics Basis
    # Y_0 = N_0 P_10^0(u)
    # Y_m = sqrt(2) N_m P_10^m(u) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(u) sin(|m| phi) for m < 0
    # N_m = sqrt( (21/(4 pi)) * (10-m)! / (10+m)! )
    
    # Precompute N_m
    N_m = np.zeros(n_coeffs)
    for m in range(-k, k + 1):
        abs_m = abs(m)
        N_m[m + k] = math.sqrt((2 * k + 1) / (4 * np.pi) * factorial(k - abs_m) / factorial(k + abs_m))
        
    # Precompute P_10^m(u) for all u and m
    # We need P_10^m(u) for m=0..10 at all u_gl points
    # scipy.special.lpmv(m, l, x) gives P_l^m(x)
    # Note: lpmv includes Condon-Shortley phase (-1)^m
    P_vals = np.zeros((len(x_gl), k + 1)) # P_vals[i, m] = P_10^m(x_gl[i])
    for m in range(k + 1):
        P_vals[:, m] = scipy.special.lpmv(m, k, x_gl)
        
    # Precompute trigonometric factors for phi
    cos_m_phi = np.zeros((n_phi, k + 1))
    sin_m_phi = np.zeros((n_phi, k + 1))
    for m in range(k + 1):
        cos_m_phi[:, m] = np.cos(m * phi_vals)
        sin_m_phi[:, m] = np.sin(m * phi_vals)
        
    # Build the basis matrix Y(u, phi)
    # Shape: (n_u, n_phi, n_coeffs)
    # Y[i, j, c_idx] is the value of the basis function for coefficient c_idx at u[i], phi[j]
    
    Y_basis = np.zeros((len(x_gl), n_phi, n_coeffs))
    
    for c_idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        if m == 0:
            # Y_0 = N_0 P_10^0(u)
            # Values are constant in phi
            val = N_m[0] * P_vals[:, 0]
            Y_basis[:, :, c_idx] = val[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            # Y_m = sqrt(2) N_m P_10^m(u) cos(m phi)
            val_u = N_m[m] * P_vals[:, m]
            val_phi = cos_m_phi[:, m]
            Y_basis[:, :, c_idx] = np.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
        else:
            # Y_m = sqrt(2) N_|m| P_10^|m|(u) sin(|m| phi)
            val_u = N_m[abs_m] * P_vals[:, abs_m]
            val_phi = sin_m_phi[:, abs_m]
            Y_basis[:, :, c_idx] = np.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
            
    # Quadrature weights for the full sphere
    # Weight for each (u, phi) point is w_gl[i] * dphi
    # Total measure is 4*pi
    w_u = w_gl
    w_phi = np.full(n_phi, dphi)
    
    # We can flatten the basis for easier vector operations
    # But it's large: 70 * 130 * 21 = 191100. Manageable.
    
    # --- Objective Function ---
    
    def compute_ratio(c):
        # c is shape (21,)
        # f = Y_basis @ c
        # f shape: (70, 130)
        f = np.tensordot(Y_basis, c, axes=([2], [0]))
        
        # ||f||_2^2 = sum w_i w_j f^2
        # ||f||_4^4 = sum w_i w_j f^4
        
        # To avoid creating huge arrays for weights, we can use einsum or simple multiplication
        # Weights are separable: W[i, j] = w_u[i] * w_phi[j]
        
        f2 = f * f
        f4 = f2 * f2
        
        # Sum over phi first, then u
        # sum_j f2[i, j] * w_phi[j]
        sum_f2_phi = np.sum(f2 * w_phi[np.newaxis, :], axis=1)
        sum_f4_phi = np.sum(f4 * w_phi[np.newaxis, :], axis=1)
        
        norm2_sq = np.sum(sum_f2_phi * w_u)
        norm4_4 = np.sum(sum_f4_phi * w_u)
        
        # Avoid division by zero
        if norm2_sq < 1e-30:
            return np.inf
            
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.power(norm4_4, 0.25)
        
        return norm4 / norm2
        
    def objective(c):
        return compute_ratio(c)
        
    # --- Optimization ---
    
    start_time = time.time()
    deadline = 620.0 # Leave 10s margin
    
    best_c = np.ones(n_coeffs)
    best_score = objective(best_c)
    
    # 1. Differential Evolution
    # Bounds: coefficients can be large, but ratio is scale invariant.
    # We can constrain ||c||_2 = 1 to remove scale invariance, but DE handles unbounded better with bounds.
    # Let's use bounds [-10, 10] for each coefficient.
    bounds = [(-10.0, 10.0)] * n_coeffs
    
    # Use a subset of time for DE
    de_time_budget = 120.0
    
    def de_callback(x, convergence):
        nonlocal best_c, best_score
        current_score = objective(x)
        if current_score < best_score:
            best_score = current_score
            best_c = x.copy()
        # Check time
        if time.time() - start_time > de_time_budget:
            return True
        return False
        
    try:
        res_de = minimize(
            objective, 
            np.zeros(n_coeffs), # Initial guess
            method='differential_evolution', 
            bounds=bounds, 
            maxiter=500, 
            popsize=15, 
            tol=1e-6, 
            update_interval=1,
            callback=de_callback,
            seed=42,
            polish=True # Use BFGS at the end of DE
        )
        if objective(res_de.x) < best_score:
            best_score = objective(res_de.x)
            best_c = res_de.x.copy()
    except Exception as e:
        print(f"DE failed: {e}")
        
    # 2. Random Restarts with BFGS
    # Use remaining time for multiple restarts
    while time.time() - start_time < deadline:
        # Random initialization
        # Scale random vector to have norm 1 to avoid huge/small values initially
        c0 = np.random.randn(n_coeffs)
        c0 /= np.linalg.norm(c0)
        
        # BFGS
        try:
            res_bfgs = minimize(
                objective, 
                c0, 
                method='BFGS', 
                options={'maxiter': 1000, 'gtol': 1e-8}
            )
            score = objective(res_bfgs.x)
            if score < best_score:
                best_score = score
                best_c = res_bfgs.x.copy()
        except Exception:
            pass
            
        # Add some noise to best_c to escape local minima?
        # Or just do more random restarts.
        
    # 3. Final Polish on best_c
    try:
        res_final = minimize(
            objective, 
            best_c, 
            method='BFGS', 
            options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-15}
        )
        score = objective(res_final.x)
        if score < best_score:
            best_score = score
            best_c = res_final.x.copy()
    except Exception:
        pass
        
    # Validation
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(n_coeffs)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Normalize to be within range
        if max_abs == 0:
            return np.ones(n_coeffs)
        best_c = best_c / max_abs * 1.0
        
    return best_c

# Need to import scipy.special inside or at top? It's at top.
# Wait, I used scipy.special.lpmv but didn't import scipy.special explicitly in the snippet above?
# Ah, I used `from scipy.special import factorial` but not lpmv.
# Let's fix imports.

# Revised Imports
import numpy as np
import time
import math
import scipy.special
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Finds the coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimizes the ratio R = ||f||_4 / ||f||_2.
    
    Strategy:
    1. Precompute quadrature nodes and weights.
    2. Use Differential Evolution for global exploration.
    3. Use BFGS for local polishing.
    4. Time-guarded to return best result before 630s.
    """
    
    k = 10
    n_coeffs = 2 * k + 1
    
    # --- Precomputation ---
    
    # 1. Gauss-Legendre nodes and weights for cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    
    # 2. Equally spaced phi points
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # 3. Spherical Harmonics Basis
    N_m = np.zeros(n_coeffs)
    for m in range(-k, k + 1):
        abs_m = abs(m)
        N_m[m + k] = math.sqrt((2 * k + 1) / (4 * np.pi) * scipy.special.factorial(k - abs_m) / scipy.special.factorial(k + abs_m))
        
    P_vals = np.zeros((len(x_gl), k + 1))
    for m in range(k + 1):
        P_vals[:, m] = scipy.special.lpmv(m, k, x_gl)
        
    cos_m_phi = np.zeros((n_phi, k + 1))
    sin_m_phi = np.zeros((n_phi, k + 1))
    for m in range(k + 1):
        cos_m_phi[:, m] = np.cos(m * phi_vals)
        sin_m_phi[:, m] = np.sin(m * phi_vals)
        
    Y_basis = np.zeros((len(x_gl), n_phi, n_coeffs))
    
    for c_idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        if m == 0:
            val = N_m[0] * P_vals[:, 0]
            Y_basis[:, :, c_idx] = val[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            val_u = N_m[m] * P_vals[:, m]
            val_phi = cos_m_phi[:, m]
            Y_basis[:, :, c_idx] = np.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
        else:
            val_u = N_m[abs_m] * P_vals[:, abs_m]
            val_phi = sin_m_phi[:, abs_m]
            Y_basis[:, :, c_idx] = np.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
            
    w_u = w_gl
    w_phi = np.full(n_phi, dphi)
    
    # --- Objective Function ---
    
    def compute_ratio(c):
        f = np.tensordot(Y_basis, c, axes=([2], [0]))
        f2 = f * f
        f4 = f2 * f2
        sum_f2_phi = np.sum(f2 * w_phi[np.newaxis, :], axis=1)
        sum_f4_phi = np.sum(f4 * w_phi[np.newaxis, :], axis=1)
        norm2_sq = np.sum(sum_f2_phi * w_u)
        norm4_4 = np.sum(sum_f4_phi * w_u)
        if norm2_sq < 1e-30:
            return np.inf
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.power(norm4_4, 0.25)
        return norm4 / norm2
        
    def objective(c):
        return compute_ratio(c)
        
    # --- Optimization ---
    
    start_time = time.time()
    deadline = 620.0
    
    best_c = np.ones(n_coeffs)
    best_score = objective(best_c)
    
    bounds = [(-10.0, 10.0)] * n_coeffs
    
    de_time_budget = 120.0
    
    def de_callback(x, convergence):
        nonlocal best_c, best_score
        current_score = objective(x)
        if current_score < best_score:
            best_score = current_score
            best_c = x.copy()
        if time.time() - start_time > de_time_budget:
            return True
        return False
        
    try:
        res_de = minimize(
            objective, 
            np.zeros(n_coeffs), 
            method='differential_evolution', 
            bounds=bounds, 
            maxiter=500, 
            popsize=15, 
            tol=1e-6, 
            update_interval=1,
            callback=de_callback,
            seed=42,
            polish=True
        )
        if objective(res_de.x) < best_score:
            best_score = objective(res_de.x)
            best_c = res_de.x.copy()
    except Exception as e:
        pass
        
    while time.time() - start_time < deadline:
        c0 = np.random.randn(n_coeffs)
        c0 /= np.linalg.norm(c0)
        try:
            res_bfgs = minimize(
                objective, 
                c0, 
                method='BFGS', 
                options={'maxiter': 1000, 'gtol': 1e-8}
            )
            score = objective(res_bfgs.x)
            if score < best_score:
                best_score = score
                best_c = res_bfgs.x.copy()
        except Exception:
            pass
            
    try:
        res_final = minimize(
            objective, 
            best_c, 
            method='BFGS', 
            options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-15}
        )
        score = objective(res_final.x)
        if score < best_score:
            best_score = score
            best_c = res_final.x.copy()
    except Exception:
        pass
        
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(n_coeffs)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            return np.ones(n_coeffs)
        best_c = best_c / max_abs * 1.0
        
    return best_c
