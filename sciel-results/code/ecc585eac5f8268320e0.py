"""
Strategy: Minimize the Ratio of Fourier L1 Norm to Function L1 Norm
The Cohn-Goncalves uncertainty bound often involves the ratio 
||f_hat||_1 / ||f||_1. The current program minimizes only ||f_hat||_1, 
which may not minimize the actual bound if ||f||_1 varies significantly.
We switch to minimizing this ratio, use asymptotic Laguerre root 
distribution for initial guesses, and employ Powell's method for 
robust derivative-free optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_laguerre
import scipy.integrate as integrate

# EVOLVE-BLOCK-START

def _compute_fourier_l1_and_norm(roots):
    """
    Computes the L1 norm of the Fourier transform and the L1 norm of the function.
    f(x) = P(x) e^{-x} for x > 0.
    P(x) = prod (x - r_i)^2.
    """
    roots = np.sort(roots)
    if np.any(roots <= 0):
        return 1e9, 1e9
        
    # 1. Compute coefficients of P(x)
    coeffs = np.array([1.0])
    for r in roots:
        new_coeffs = np.zeros(len(coeffs) + 2)
        new_coeffs[2:] += coeffs
        new_coeffs[1:-1] += -2.0 * r * coeffs
        new_coeffs[:-2] += r**2 * coeffs
        coeffs = new_coeffs
        
    # 2. Compute ||f||_1 = sum c_k k!
    fact = np.array([1.0])
    for i in range(1, len(coeffs)):
        fact = np.append(fact, fact[-1] * i)
        
    l1_norm_f = np.sum(coeffs * fact)
    if l1_norm_f <= 0:
        return 1e9, 1e9

    # 3. Compute ||f_hat||_1
    # F(xi) = sum c_k k! / (1 + i xi)^(k+1)
    # We integrate |F(xi)| from -inf to inf.
    # F(xi) is even in magnitude? 
    # F(-xi) = sum c_k k! / (1 - i xi)^(k+1) = conj(F(xi)) if c_k are real.
    # So |F(-xi)| = |F(xi)|.
    # We can integrate from 0 to inf and multiply by 2.
    
    # Grid for xi
    # The decay is 1/xi^25. 
    # We need to capture the peak at 0.
    # Use a log-spaced grid for the tail? 
    # Or a fine linear grid near 0 and coarse far away.
    # Let's use a composite grid.
    
    xi_near = np.linspace(0, 10.0, 10000)
    xi_far = np.linspace(10.0, 100.0, 1000)
    xi = np.concatenate([xi_near, xi_far])
    
    denom_base = 1.0 + 1j * xi
    F = np.zeros_like(xi, dtype=complex)
    
    for k, c in enumerate(coeffs):
        if c == 0.0: continue
        F += c * fact[k] / (denom_base ** (k + 1))
        
    # Integrate |F(xi)| from 0 to inf
    # Use trapz
    integral_half = np.trapz(np.abs(F), xi)
    l1_norm_fhat = 2.0 * integral_half
    
    return l1_norm_fhat, l1_norm_f

def _objective(params):
    """
    Objective: Minimize ||f_hat||_1 / ||f||_1
    """
    roots = np.sort(params)
    if np.any(roots <= 0) or np.any(np.diff(roots) < 1e-6):
        return 1e9
        
    try:
        l1_fhat, l1_f = _compute_fourier_l1_and_norm(roots)
        if l1_f == 0:
            return 1e9
        return l1_fhat / l1_f
    except Exception:
        return 1e9

def construct():
    deadline = time.time() + 630.0 - 60.0  # 570s safety margin
    n_roots = 12
    
    # Asymptotic Laguerre roots: x_k approx 4n sin^2(k pi / (2n+2))
    n = 12
    k = np.arange(1, n_roots + 1)
    initial_roots = 4 * n * np.sin(np.pi * k / (2 * n + 2))**2
    
    best_roots = initial_roots.copy()
    best_score = _objective(best_roots)
    
    # Multiple starts with Powell's Method
    rng = np.random.default_rng(42)
    
    # Generate 10 perturbations
    starts = [initial_roots]
    for _ in range(10):
        perturbed = initial_roots * (1 + rng.normal(0, 0.1, n_roots))
        perturbed = np.sort(perturbed)
        perturbed = np.maximum(perturbed, 0.1)
        starts.append(perturbed)
        
    # Also include the uniform guess from the original program
    uniform_roots = np.arange(1, n_roots + 1) * 4.0
    starts.append(uniform_roots)
    
    for start in starts:
        if time.time() > deadline - 30.0:
            break
            
        try:
            # Powell's method is derivative-free and robust
            # Bounds are not directly supported in Powell, but we can clip
            # or use a transformation. 
            # Since roots must be positive and increasing, we optimize the gaps.
            # Let r_1 = g_0, r_2 = g_0 + g_1, ...
            # g_i > 0.
            # This ensures ordering and positivity.
            
            def _gap_objective(gaps):
                roots = np.cumsum(gaps)
                if np.any(gaps <= 0):
                    return 1e9
                return _objective(roots)
            
            # Initial gaps
            gaps = np.diff(np.sort(start))
            gaps = np.insert(gaps, 0, start[0])
            
            res = minimize(
                _gap_objective,
                gaps,
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_roots = np.cumsum(res.x)
                
        except Exception:
            pass
            
    # Final polish with Nelder-Mead on the roots directly?
    # Powell on gaps is good.
    
    # Ensure final roots are valid
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    for i in range(1, len(best_roots)):
        if best_roots[i] <= best_roots[i-1]:
            best_roots[i] = best_roots[i-1] + 1e-6
            
    return [float(r) for r in best_roots]

# EVOLVE-BLOCK-END
