"""
Strategy: Direct L1-Norm Minimization of the Fourier Transform
Instead of a surrogate for root spacing, we directly minimize the L1 norm
of the Fourier transform of the Cohn-Goncalves Laguerre polynomial.
This is the quantity that controls the sign uncertainty bound.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_laguerre
import scipy.integrate as integrate

# EVOLVE-BLOCK-START

def _compute_fourier_l1(roots, freq_max=100.0, n_points=20000):
    """
    Computes the L1 norm of the Fourier transform of the Cohn-Goncalves
    Laguerre polynomial constructed from the given double roots.
    
    The polynomial is P(x) = prod (x - r_i)^2.
    The function is f(x) = P(x) * exp(-x/2) * x^{n-1/2} ? 
    Actually, standard Laguerre L_n(x) = e^x / n! d^n/dx^n (e^{-x} x^n).
    The Cohn-Goncalves construction uses a specific linear combination.
    However, for m double roots, it is often modeled as a single polynomial 
    with double roots weighted by the Laguerre weight.
    
    Let's use the standard Fourier transform definition for the weighted polynomial.
    f(x) = P(x) * e^{-x} for x > 0, 0 otherwise.
    F(xi) = integral_0^inf P(x) e^{-x} e^{-i xi x} dx.
    
    We can compute this integral numerically. To be fast, we pre-compute
    the moments or use a fixed quadrature.
    """
    # Sort roots to ensure stability
    roots = np.sort(roots)
    
    # Define the polynomial P(x) = prod (x - r_i)^2
    # We need to integrate P(x) e^{-x} e^{-i xi x}
    
    # To avoid computing the polynomial explicitly (which can overflow),
    # we can use the fact that for double roots, the function is smooth.
    # However, with 12 double roots, the degree is 24.
    # P(x) is a polynomial of degree 24.
    
    # Let's use a fixed grid for x and xi.
    # x grid: 0 to x_max where P(x)e^{-x} is negligible.
    # xi grid: -freq_max to freq_max.
    
    # Optimization: We can compute the Fourier transform analytically 
    # if we expand P(x) into coefficients.
    # P(x) = sum c_k x^k.
    # Integral x^k e^{-x} e^{-i xi x} = Gamma(k+1) / (1 + i xi)^{k+1}.
    # This is MUCH faster and more stable than numerical integration.
    
    # 1. Compute coefficients of P(x) = prod (x - r_i)^2
    # Start with [1]
    coeffs = np.array([1.0])
    for r in roots:
        # (x - r)^2 = x^2 - 2rx + r^2
        # Multiply current coeffs by (x^2 - 2rx + r^2)
        # coeffs is [c0, c1, ..., cn] for c0 + c1 x + ... + cn x^n
        # New coeffs will have degree n+2
        new_coeffs = np.zeros(len(coeffs) + 2)
        
        # x^2 * coeffs -> shift by 2
        new_coeffs[2:] += coeffs
        # -2r x * coeffs -> shift by 1, scale by -2r
        new_coeffs[1:-1] += -2.0 * r * coeffs
        # r^2 * coeffs -> no shift, scale by r^2
        new_coeffs[:-2] += r**2 * coeffs
        
        coeffs = new_coeffs
    
    # coeffs[k] is the coefficient of x^k.
    # Degree is 24.
    
    # 2. Compute F(xi) = sum_{k=0}^{24} c_k * k! / (1 + i xi)^{k+1}
    # We evaluate this on a grid of xi.
    
    # Choose xi grid. The decay is ~ 1/|xi|^{25}.
    # We need to capture the peak at xi=0.
    # Let's use a log-spaced or linear grid. Linear is safer for L1.
    # Range: The roots are around 4..48. The frequency scale is ~ 1/root.
    # Let's use xi from -100 to 100.
    
    xi = np.linspace(-100.0, 100.0, 40000)
    
    # Precompute factorials
    fact = np.array([1.0])
    for i in range(1, len(coeffs)):
        fact = np.append(fact, fact[-1] * i)
    
    # F(xi) is complex
    F = np.zeros_like(xi, dtype=complex)
    
    # Vectorized computation
    # (1 + i xi) is a complex vector
    denom_base = 1.0 + 1j * xi
    
    for k, c in enumerate(coeffs):
        if c == 0.0: continue
        # term = c * k! / denom_base^(k+1)
        # To avoid overflow/underflow in powers, we can compute iteratively
        # or use np.power.
        F += c * fact[k] / (denom_base ** (k + 1))
        
    # L1 norm is integral |F(xi)| dxi
    # Use trapezoidal rule
    l1_norm = np.trapz(np.abs(F), xi)
    
    return l1_norm

def _objective(params):
    """
    Objective function for optimization.
    params: array of 12 positive root locations.
    """
    roots = np.sort(params)
    # Ensure strictly positive and increasing
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        return 1e9
        
    try:
        val = _compute_fourier_l1(roots)
        return val
    except Exception:
        return 1e9

def construct():
    deadline = time.time() + 630.0 - 60.0  # 570s safety margin
    
    # Initial guess: Arithmetic progression 4, 8, ..., 48
    n_roots = 12
    initial_roots = np.arange(1, n_roots + 1) * 4.0
    
    best_roots = initial_roots.copy()
    best_score = _objective(best_roots)
    
    # 1. Global Search: Differential Evolution
    # Bounds: Roots must be positive and ordered.
    # We optimize the gaps to ensure ordering.
    # Let g_i be the gap before root i. root_1 = g_0, root_2 = g_0 + g_1, etc.
    # Simpler: Optimize roots directly with bounds, then sort.
    # But DE doesn't handle ordering constraints well.
    # Let's optimize the roots directly and sort them in the objective.
    # Bounds: [1, 100] for each root.
    
    bounds = [(1.0, 100.0)] * n_roots
    
    # Run DE with a limited number of iterations to stay within budget
    # DE is slow, so we use a small population and maxiter.
    try:
        if time.time() < deadline - 300.0: # Allow 300s for DE
            res_de = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                update_interval=5,
                polish=False # We will polish manually
            )
            if res_de.success or res_de.fun < best_score:
                best_roots = res_de.x
                best_score = res_de.fun
    except Exception:
        pass
    
    # 2. Local Search: L-BFGS-B
    # Multiple starts from DE result and initial guess
    starts = [best_roots, initial_roots]
    
    # Add some random perturbations to initial roots
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbed = initial_roots + rng.normal(0, 2.0, n_roots)
        perturbed = np.sort(perturbed)
        perturbed = np.maximum(perturbed, 1.0)
        starts.append(perturbed)
        
    for start in starts:
        if time.time() > deadline - 60.0:
            break
            
        try:
            res_lbfgs = minimize(
                _objective,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if res_lbfgs.fun < best_score:
                best_score = res_lbfgs.fun
                best_roots = res_lbfgs.x
        except Exception:
            pass
            
    # 3. Final Polish with Nelder-Mead for robustness
    if time.time() < deadline - 30.0:
        try:
            res_nm = minimize(
                _objective,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_roots = res_nm.x
        except Exception:
            pass

    # Ensure the final roots are valid (positive, increasing)
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    # Enforce strict increase
    for i in range(1, len(best_roots)):
        if best_roots[i] <= best_roots[i-1]:
            best_roots[i] = best_roots[i-1] + 1e-6
            
    return [float(r) for r in best_roots]

# EVOLVE-BLOCK-END
