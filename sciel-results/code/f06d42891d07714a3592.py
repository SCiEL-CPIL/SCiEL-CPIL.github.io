import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 180s limit
    
    # Hermite polynomials (physicists' convention)
    def H0(x):
        return np.ones_like(x, dtype=float)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def poly(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def f_of_x(x, c0, c4, c8):
        return poly(x, c0, c4, c8) * np.exp(-x**2 / 2.0)
    
    def f_prime_of_x(x, c0, c4, c8):
        # Derivative of P(x)*exp(-x^2/2) = P'(x)*exp(-x^2/2) - x*P(x)*exp(-x^2/2)
        # = [P'(x) - x*P(x)] * exp(-x^2/2)
        dx = 1e-8
        return (f_of_x(x+dx, c0, c4, c8) - f_of_x(x-dx, c0, c4, c8)) / (2*dx)
    
    def compute_uncertainty_product(coeffs):
        c0, c4, c8 = coeffs
        
        # Check time
        if time.time() > deadline - 5:
            return np.inf
        
        try:
            # Compute ∫|f(x)|² dx
            norm_sq, _ = integrate.quad(lambda x: f_of_x(x, c0, c4, c8)**2, -np.inf, np.inf, limit=200)
            if norm_sq <= 0 or not np.isfinite(norm_sq):
                return np.inf
            
            # Compute ∫x²|f(x)|² dx
            x2_norm, _ = integrate.quad(lambda x: x**2 * f_of_x(x, c0, c4, c8)**2, -np.inf, np.inf, limit=200)
            if not np.isfinite(x2_norm) or x2_norm < 0:
                return np.inf
            
            # σ_x² = ∫x²|f|²dx / ∫|f|²dx
            sigma_x2 = x2_norm / norm_sq
            
            # For frequency: σ_ξ² = (1/(4π²)) * ∫(f'(x))²dx / ∫|f(x)|²dx
            # This comes from: ∫ξ²|F̂(f)(ξ)|²dξ = (1/(4π²))∫(f'(x))²dx
            # with F̂(f)(ξ) = ∫f(x)e^{-2πixξ}dx
            
            fprime_sq, _ = integrate.quad(lambda x: f_prime_of_x(x, c0, c4, c8)**2, -np.inf, np.inf, limit=200)
            if not np.isfinite(fprime_sq) or fprime_sq < 0:
                return np.inf
            
            sigma_xi2 = fprime_sq / (4 * np.pi**2 * norm_sq)
            
            # Uncertainty product: σ_x * σ_ξ
            product = np.sqrt(sigma_x2 * sigma_xi2)
            
            return product
            
        except Exception:
            return np.inf
    
    # Objective: minimize the uncertainty product σ_x * σ_ξ
    def objective(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        return compute_uncertainty_product(coeffs)
    
    # Initial guess - try various starting points
    x0 = np.array([0.0, 0.0, 0.0])
    
    best_score = objective(x0)
    best_coeffs = x0.copy()
    
    # Try multiple initial guesses
    initial_guesses = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, -0.1, 0.01]),
        np.array([-1.0, 0.1, -0.01]),
        np.array([0.5, -0.05, 0.005]),
        np.array([2.0, -0.2, 0.02]),
        np.array([-0.5, 0.05, -0.005]),
        np.array([1.0, 0.1, 0.01]),
        np.array([-1.0, -0.1, -0.01]),
    ]
    
    rng = np.random.RandomState(42)
    
    # Evaluate initial guesses
    for guess in initial_guesses:
        if time.time() > deadline - 15:
            break
        score = objective(guess)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_coeffs = guess.copy()
    
    # Nelder-Mead optimization from multiple restarts
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        if restart < len(initial_guesses):
            x_start = initial_guesses[restart]
        elif restart < len(initial_guesses) + 10:
            # Random restarts around best
            scale = 0.3 * (1 + (restart - len(initial_guesses))/5.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
        else:
            # Wider random restarts
            x_start = rng.normal(0, 2.0, 3)
        
        try:
            res = opt.minimize(objective, x_start, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
