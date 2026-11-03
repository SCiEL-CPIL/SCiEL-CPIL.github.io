"""
Optimized program using a different structural approach:
1. Pre-compute all analytical constants for the bound formula
2. Use a multi-start local optimization with Nelder-Mead simplex method
3. Add a fine-grained grid search around the seed point
4. Use the Nelder-Mead method which is derivative-free and often finds
   better local minima than L-BFGS-B for this type of problem
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Pre-compute constants for the bound formula
# H0 = 1
# H4 = 16x^4 - 48x^2 + 12
# H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330

# Coefficients for P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
# where c12 = -(c0 + 12*c4 + 168*c8) / 1330
# P(x) = a12*x^12 + a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0

# For the integral computations, we need:
# integral of x^(2k) * exp(-x^2) dx from -inf to inf = sqrt(pi) * 2^k * k!

def _precompute_integrals(max_degree=30):
    """Pre-compute the Gaussian moment integrals."""
    integrals = np.zeros(max_degree + 1)
    for k in range(max_degree + 1):
        integrals[k] = math.sqrt(math.pi) * (2 ** k) * math.factorial(k)
    return integrals

# Pre-compute the integrals once
_INTEGRALS = _precompute_integrals(30)

def _compute_bound_fast(c0, c4, c8):
    """
    Fast computation of the uncertainty product bound.
    Uses pre-computed constants for speed.
    """
    c12 = -(c0 + 12 * c4 + 168 * c8) / 1330.0
    
    # Coefficients of P(x)
    a0 = c0 + 12 * c4 + 168 * c8 + 1330 * c12
    a2 = -48 * c4 - 1344 * c8 - 32256 * c12
    a4 = 16 * c4 + 2688 * c8 + 133056 * c12
    a6 = -1344 * c8 - 180224 * c12
    a8 = 256 * c8 + 118272 * c12
    a10 = -36864 * c12
    a12 = 4096 * c12
    
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Convolution to get coefficients of P(x)^2
    q = np.convolve(p, p)
    
    # Integrate x^(2k) e^{-x^2}
    # q[k] corresponds to x^(2k) term
    I = q * _INTEGRALS[:len(q)]
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10
    
    x2 = I[1] / I[0]
    
    # Calculate R(x) for Fourier side
    # R(x) = sum of terms from the derivative structure
    r = np.zeros(8)
    for j in range(8):
        val = 0.0
        if j <= 5:
            val += 2 * (j + 1) * (2 * j + 1) * p[j + 1]
        if j >= 1 and j <= 6:
            val += -4 * j * p[j]
        if j >= 1 and j <= 7:
            val += p[j - 1]
        if j <= 6:
            val += -p[j]
        r[j] = val
    
    s = np.convolve(r, r)
    
    I_R = s * _INTEGRALS[:len(s)]
    norm_R_sq = I_R[0]
    
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    
    product = math.sqrt(x2 * xi2)
    
    return product

def _nelder_mead_minimize(objective, x0, bounds, maxiter=500, tol=1e-12):
    """
    Custom Nelder-Mead implementation with bounds handling.
    Uses reflection, expansion, and contraction steps.
    """
    n = len(x0)
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = x0.copy()
        step = 0.05 if i != 0 else 0.1
        point[i] += step
        # Clip to bounds
        for j in range(n):
            point[j] = np.clip(point[j], bounds[j][0], bounds[j][1])
        simplex.append(point)
    
    # Evaluate initial simplex
    values = [objective(point) for point in simplex]
    
    # Track best
    best_idx = np.argmin(values)
    best_point = simplex[best_idx].copy()
    best_value = values[best_idx]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    for iteration in range(maxiter):
        # Sort simplex by function values
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if values[-1] - values[0] < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        # Clip to bounds
        for j in range(n):
            xr[j] = np.clip(xr[j], bounds[j][0], bounds[j][1])
        fr = objective(xr)
        
        if fr < values[0]:
            # Try expansion
            xe = centroid + gamma * (xr - centroid)
            for j in range(n):
                xe[j] = np.clip(xe[j], bounds[j][0], bounds[j][1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            # Accept reflection
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Try contraction
            if fr < values[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (xr - centroid)
            for j in range(n):
                xc[j] = np.clip(xc[j], bounds[j][0], bounds[j][1])
            fc = objective(xc)
            
            if fc < min(fr, values[-1]):
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    for j in range(n):
                        simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
                    values[i] = objective(simplex[i])
        
        # Update best
        current_best_idx = np.argmin(values)
        if values[current_best_idx] < best_value:
            best_value = values[current_best_idx]
            best_point = simplex[current_best_idx].copy()
    
    return best_point, best_value

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = _compute_bound_fast(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    # Define bounds for the search
    bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]
    
    # Objective function wrapper
    def objective(x):
        try:
            return _compute_bound_fast(x[0], x[1], x[2])
        except Exception:
            return 1e10
    
    # Multi-start Nelder-Mead optimization
    # Generate multiple starting points
    start_points = [seed_coeffs]
    
    # Add points around the seed
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        for i in range(3):
            offset = np.random.randn(3) * scale
            point = seed_coeffs + offset
            # Clip to bounds
            for j in range(3):
                point[j] = np.clip(point[j], bounds[j][0], bounds[j][1])
            start_points.append(point)
    
    # Also try some random points
    for _ in range(10):
        point = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        start_points.append(point)
    
    # Run Nelder-Mead from each starting point
    for point in start_points:
        if time.time() > deadline - 5.0:
            break
        
        try:
            result_point, result_value = _nelder_mead_minimize(
                objective, point, bounds, maxiter=300, tol=1e-12
            )
            if result_value < best_val:
                best_val = result_value
                best_coeffs = result_point
        except Exception:
            pass
    
    # Fine-tuning with scipy's Nelder-Mead around the best point
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    # Try COBYLA as well for a different local optimization landscape
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(
                objective,
                best_coeffs,
                method='COBYLA',
                bounds=bounds,
                options={'maxiter': 1000, 'tol': 1e-12}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_coeffs = res2.x
        except Exception:
            pass
    
    # Final polish with L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            res3 = minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res3.fun < best_val:
                best_val = res3.fun
                best_coeffs = res3.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
