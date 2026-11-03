import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.special import eval_hermite
import time
import sympy

# Pre-compute Hermite values and weights for fast objective evaluation
# Use Gauss-Hermite quadrature for high precision and speed
# We need integrals of form int P(x)^k exp(-a x^2) dx
# Let's use a fixed dense grid with trapezoidal rule for simplicity and robustness,
# but pre-compute the basis functions and weights.

# Grid for numerical integration
# The Gaussian decays fast. [-5, 5] with 2000 points should be very accurate.
# Let's use [-6, 6] with 4000 points to be safe, or Gauss-Hermite if possible.
# scipy.integrate.quad_vec is slow. Fixed grid is faster for repeated calls.

_X_MIN = -8.0
_X_MAX = 8.0
_N_PTS = 8000
_x_grid = np.linspace(_X_MIN, _X_MAX, _N_PTS)
_dx = _x_grid[1] - _x_grid[0]

# Pre-compute Hermite polynomials on the grid
# H_n(x)
_H0 = np.ones_like(_x_grid)
_H4 = eval_hermite(4, _x_grid)
_H8 = eval_hermite(8, _x_grid)
_H12 = eval_hermite(12, _x_grid)

# Pre-compute exponentials
# Note: The objective uses exp(-2 pi x^2) for norm2 and exp(-4 pi x^2) for norm4.
# Wait, the task description says "Gaussian test function".
# Usually g(x) = P(x) e^{-pi x^2}.
# Then |g|^2 = P^2 e^{-2 pi x^2}.
# |g|^4 = P^4 e^{-4 pi x^2}.
# The Fourier transform of e^{-pi x^2} is e^{-pi xi^2}.
# So the Fourier side has the same form with P_minus.

_exp_2pi = np.exp(-2 * np.pi * _x_grid**2)
_exp_4pi = np.exp(-4 * np.pi * _x_grid**2)

# Weights for integration
_w_2 = _exp_2pi * _dx
_w_4 = _exp_4pi * _dx

def _compute_objective_fast(coeffs):
    return _compute_objective_fast_impl(coeffs)

def _compute_objective_fast_impl(coeffs):
    a0, a4, a8 = coeffs
    
    # Calculate a12 such that P(0) = 0
    # P(0) = a0*1 + a4*(-12) + a8*(1680) + a12*(-332640) = 0
    # a12 = (a0 - 12*a4 + 1680*a8) / 332640
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 332640.0
    
    # Compute P(x) and P_minus(x)
    # P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # P_minus = a0*H0 + a4*H4 + a8*H8 - a12*H12
    
    # We can compute the integrals efficiently using dot products
    # norm2_sq = int P^2 e^{-2pi x^2} dx
    # norm4_sq = int P^4 e^{-4pi x^2} dx
    
    # To avoid allocating large arrays for P and P_minus in every call,
    # we can compute the powers directly or use the fact that we only need 3 coeffs.
    # However, with 4000 points, allocation is cheap.
    
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    P_minus = a0*_H0 + a4*_H4 + a8*_H8 - a12*_H12
    
    # Squares and 4th powers
    P2 = P * P
    P4 = P2 * P2
    
    Pm2 = P_minus * P_minus
    Pm4 = Pm2 * Pm2
    
    norm2_sq = np.dot(P2, _w_2)
    norm4_sq = np.dot(P4, _w_4)
    
    norm2_sq_hat = np.dot(Pm2, _w_2)
    norm4_sq_hat = np.dot(Pm4, _w_4)
    
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
        
    obj = (norm4_sq / (norm2_sq * norm2_sq)) * (norm4_sq_hat / (norm2_sq_hat * norm2_sq_hat))
    
    return obj

def construct():
    start_time = time.time()
    deadline = 180.0
    
    # Prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    seed_score = _compute_objective_fast(x0)
    
    best_result = x0
    best_score = seed_score
    
    # Multi-start local optimization
    # Use L-BFGS-B with tight tolerances
    # Generate diverse starting points
    # The seed is a good starting point.
    # Let's add perturbations and some random starts within a reasonable range.
    
    np.random.seed(42)
    
    starts = [x0]
    
    # Small perturbations
    for i in range(10):
        starts.append(x0 + np.random.randn(3) * np.array([0.1, 0.01, 0.0001]))
        
    # Larger perturbations
    for i in range(10):
        starts.append(x0 + np.random.randn(3) * np.array([1.0, 0.1, 0.001]))
        
    # Specific known points or heuristics
    # Maybe try to balance the terms more?
    starts.append(np.array([-1.0, 0.1, 0.0]))
    starts.append(np.array([-0.5, 0.05, 0.0]))
    starts.append(np.array([-2.0, 0.2, 0.0]))
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([-1.13, 0.04, 0.0003]))
    
    # First pass: local optimization from multiple starts
    for s in starts:
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(_compute_objective_fast, s, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    # Second pass: Basin hopping for global exploration
    if time.time() < deadline - 20:
        try:
            def basin_obj(x):
                return _compute_objective_fast(x)
                
            res_bh = basinhopping(basin_obj, x0=best_result, 
                                  minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 500}},
                                  niter=20, stepsize=0.5, seed=42)
            if res_bh.fun < best_score:
                best_score = res_bh.fun
                best_result = res_bh.x
        except Exception:
            pass
            
    # Third pass: Differential evolution for broader search if time permits
    if time.time() < deadline - 15:
        try:
            # Define bounds for differential evolution
            # Based on the scale of the seed coefficients
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
            
            res_de = differential_evolution(_compute_objective_fast, bounds, 
                                            maxiter=100, popsize=15, seed=42, 
                                            tol=1e-12, polish=True)
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_result = res_de.x
        except Exception:
            pass
            
    # KKT symbolic search: find all stationary points exactly
    if time.time() < deadline - 40:
        try:
            kkt_solutions = _kkt_symbolic_search()
            for sol in kkt_solutions:
                try:
                    # Refine with local optimization
                    res_kkt = minimize(_compute_objective_fast, sol, method='L-BFGS-B',
                                      options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
                    if res_kkt.fun < best_score:
                        best_score = res_kkt.fun
                        best_result = res_kkt.x
                except Exception:
                    pass
        except Exception:
            pass
            
    # Final polish with even tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            res = minimize(_compute_objective_fast, best_result, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    # Ensure we return a valid result
    if best_result is None or not np.all(np.isfinite(best_result)):
        return x0
        
    return best_result

def _kkt_symbolic_search():
    """Use sympy to find stationary points of the objective via KKT conditions.
    
    The objective is f = (I4/I2^2) * (I4_hat/I2_hat^2) where:
    - I2 = ∫ P(x)^2 e^{-2πx²} dx
    - I4 = ∫ P(x)^4 e^{-4πx²} dx
    - P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    - P_minus(x) = a0*H0 + a4*H4 + a8*H8 - a12*H12
    - a12 = (a0 - 12*a4 + 1680*a8) / 332640
    
    We compute the gradient of log(f) = log(I4) - 2*log(I2) + log(I4_hat) - 2*log(I2_hat)
    and set it to zero.
    """
    try:
        a0, a4, a8 = sympy.symbols('a0 a4 a8', real=True)
        x = sympy.symbols('x', real=True)
        
        # Hermite polynomials (physicists' convention)
        def H(n, x):
            return sympy.hermite(n, x)
        
        H0 = H(0, x)
        H4 = H(4, x)
        H8 = H(8, x)
        H12 = H(12, x)
        
        # a12 from constraint P(0) = 0
        # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-332640
        a12 = (a0 - 12*a4 + 1680*a8) / 332640
        
        P = a0*H0 + a4*H4 + a8*H8 + a12*H12
        P_minus = a0*H0 + a4*H4 + a8*H8 - a12*H12
        
        # Compute integrals symbolically
        # I2 = ∫ P^2 * exp(-2*pi*x^2) dx from -∞ to ∞
        # I4 = ∫ P^4 * exp(-4*pi*x^2) dx from -∞ to ∞
        
        # Use sympy's integrate
        I2 = sympy.integrate(P**2 * sympy.exp(-2*sympy.pi*x**2), (x, -sympy.oo, sympy.oo))
        I4 = sympy.integrate(P**4 * sympy.exp(-4*sympy.pi*x**2), (x, -sympy.oo, sympy.oo))
        
        I2_hat = sympy.integrate(P_minus**2 * sympy.exp(-2*sympy.pi*x**2), (x, -sympy.oo, sympy.oo))
        I4_hat = sympy.integrate(P_minus**4 * sympy.exp(-4*sympy.pi*x**2), (x, -sympy.oo, sympy.oo))
        
        # Objective: f = (I4/I2^2) * (I4_hat/I2_hat^2)
        # log(f) = log(I4) - 2*log(I2) + log(I4_hat) - 2*log(I2_hat)
        # Gradient of log(f) w.r.t. a0, a4, a8 should be zero at stationary points
        
        log_f = sympy.log(I4) - 2*sympy.log(I2) + sympy.log(I4_hat) - 2*sympy.log(I2_hat)
        
        # Compute partial derivatives
        dlogf_da0 = sympy.diff(log_f, a0)
        dlogf_da4 = sympy.diff(log_f, a4)
        dlogf_da8 = sympy.diff(log_f, a8)
        
        # Solve the system of 3 equations
        # This may be expensive; use a timeout-like approach
        equations = [dlogf_da0, dlogf_da4, dlogf_da8]
        
        # Try to solve
        solutions = sympy.solve(equations, [a0, a4, a8], dict=True)
        
        # Filter for real, finite solutions
        valid_solutions = []
        for sol in solutions:
            try:
                vals = [sol.get(var, 0) for var in [a0, a4, a8]]
                vals_numeric = [complex(v.evalf()) for v in vals]
                # Check if real
                if all(abs(v.imag) < 1e-10 for v in vals_numeric):
                    real_vals = [v.real for v in vals_numeric]
                    if all(np.isfinite(v) for v in real_vals):
                        valid_solutions.append(np.array(real_vals, dtype=float))
            except Exception:
                continue
        
        return valid_solutions
        
    except Exception:
        return []
