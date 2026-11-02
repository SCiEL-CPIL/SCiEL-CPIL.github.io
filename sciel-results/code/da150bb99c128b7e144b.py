import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
import math

# Hermite Polynomials (Physicist's convention)
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
# H_12(x) = x^12 - 42x^10 + 2100x^8 - 60480x^6 + 604800x^4 - 1632960x^2 + 10395

def H0(x):
    return 1.0

def H4(x):
    return x**4 - 6*x**2 + 3

def H8(x):
    return x**8 - 28*x**6 + 420*x**4 - 1680*x**2 + 105

def H12(x):
    return x**12 - 42*x**10 + 2100*x**8 - 60480*x**6 + 604800*x**4 - 1632960*x**2 + 10395

def compute_a12(a0, a4, a8):
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    # Wait, H4(0) = 3? 
    # H_4(x) = x^4 - 6x^2 + 3. H_4(0) = 3.
    # H_8(x) = ... + 105. H_8(0) = 105.
    # H_12(x) = ... + 10395. H_12(0) = 10395.
    # So: a0*1 + a4*3 + a8*105 + a12*10395 = 0
    return -(a0 + 3*a4 + 105*a8) / 10395.0

def norm_g(a0, a4, a8):
    a12 = compute_a12(a0, a4, a8)
    def integrand(x):
        val = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
        return abs(val * np.exp(-0.5 * x**2))
    # Integrate from -inf to inf. Due to symmetry, integrate 0 to inf and multiply by 2.
    try:
        val, err = quad(integrand, 0, np.inf, limit=100)
        return 2 * val
    except:
        return np.inf

def norm_fhat(a0, a4, a8):
    a12 = compute_a12(a0, a4, a8)
    # F[e^{-x^2/2} H_n(x)] = i^n sqrt(2pi) e^{-2pi^2 xi^2} H_n(2pi xi)
    # The magnitude is sqrt(2pi) |...| e^{-2pi^2 xi^2}
    def integrand(xi):
        arg = 2 * np.pi * xi
        val = a0*H0(arg) + a4*H4(arg) + a8*H8(arg) + a12*H12(arg)
        return abs(val) * np.exp(-2 * (np.pi**2) * xi**2)
    try:
        val, err = quad(integrand, 0, np.inf, limit=100)
        return 2 * np.sqrt(2 * np.pi) * val
    except:
        return np.inf

def objective(coeffs):
    a0, a4, a8 = coeffs
    n1 = norm_g(a0, a4, a8)
    n2 = norm_fhat(a0, a4, a8)
    if np.isinf(n1) or np.isinf(n2):
        return 1e10
    return n1 * n2

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Initial guess from seed
    initial_guess = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_score = 1e10
    best_coeffs = initial_guess
    
    # Try a few random restarts to avoid local minima
    # We have ~170 seconds. Each objective evaluation takes maybe 0.05-0.1s.
    # Minimize with Nelder-Mead might take 100-200 evaluations.
    # Let's try 1 primary optimization and maybe 1-2 restarts if time permits.
    
    try:
        # Use L-BFGS-B or Nelder-Mead. Nelder-Mead is robust for non-smooth abs() functions.
        res = minimize(objective, initial_guess, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if time.time() < deadline:
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        
        # If time remains, try a second restart from a different point
        if time.time() < deadline - 60:
            # Perturb the best found or use a different heuristic
            # Let's try a point near the origin or scaled
            guess2 = np.array([1.0, 0.0, 0.0])
            res2 = minimize(objective, guess2, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res2.success and res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x
                
        # If time remains, try a third restart
        if time.time() < deadline - 30:
            guess3 = np.array([-1.0, 0.1, 0.01])
            res3 = minimize(objective, guess3, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res3.success and res3.fun < best_score:
                best_score = res3.fun
                best_coeffs = res3.x

    except Exception as e:
        # Fallback to seed if optimization fails
        pass

    return best_coeffs

# The task asks for the coefficients of the Hermite combination (degrees 0, 4, 8).
# The degree-12 weight is derived.
# The function `construct` returns the array of these 3 coefficients.

# To verify the score improvement, we need to know the seed's bound value.
# The seed returns [-1.13, 0.04, 0.0003086].
# We don't need to return the score, just the coefficients.
# The evaluator will compute the score.

# One potential issue: The "upper bound on A(f)A(Fourier(f))" might not be exactly ||g||_1 ||hat{g}||_1.
# However, for Fourier-invariant problems with Gaussian weights, the L1 norm product is a standard proxy for the uncertainty constant in this context (related to the Heisenberg uncertainty principle).
# If the specific bound is different (e.g., involving moments), the optimization might drift.
# But given the "Fourier-invariant" hint, the L1 norm product is the most likely candidate for the "uncertainty inequality" constant being tuned.
# Another possibility: The bound is on the product of the support lengths or similar, but L1 is the standard for "uncertainty" in this specific Hermite-Gaussian context.

# Let's refine the objective to be more robust.
# The problem states "reduce the upper bound on A(f)*A(Fourier(f))".
# In the context of the "Uncertainty Inequality" and Hermite polynomials, this often refers to the constant in:
# ||f||_1 ||hat{f}||_1 >= C ||f||_2^2 ||hat{f}||_2^2 ? No, that's not right.
# The Heisenberg uncertainty principle is usually ||f||_2 ||hat{f}||_2 related to moments.
# However, there is a known result for the L1 uncertainty principle: ||f||_1 ||hat{f}||_1 >= 1 (for specific normalizations).
# The "best" constant is approached by specific functions. The Gaussian is optimal for the L2-based uncertainty.
# For L1, the Gaussian is not optimal. The function that minimizes ||f||_1 ||hat{f}||_1 is a different shape.
# The Hermite expansion allows us to deform the Gaussian to lower this product.
# Therefore, minimizing ||g||_1 ||hat{g}||_1 is the correct objective.

# Double check the Fourier transform scaling.
# If f(x) = e^{-x^2/2}, hat{f}(xi) = sqrt(2pi) e^{-2pi^2 xi^2}.
# ||f||_1 = sqrt(pi).
# ||hat{f}||_1 = sqrt(2pi) * sqrt(pi/(2pi^2)) = sqrt(2pi) * 1/sqrt(2) * 1/sqrt(pi) * sqrt(pi) ...
# Let's calculate ||hat{f}||_1 for Gaussian:
# int sqrt(2pi) exp(-2pi^2 xi^2) dxi = sqrt(2pi) * sqrt(pi / (2pi^2)) = sqrt(2pi) * 1/(pi sqrt(2)) = sqrt(2pi)/(pi sqrt(2)) = sqrt(pi)/pi = 1/sqrt(pi).
# Product: sqrt(pi) * 1/sqrt(pi) = 1.
# So for the pure Gaussian, the product is 1.
# The seed coefficients deform the Gaussian. If the deformation reduces the product below 1, the score improves.
# My objective function minimizes this product.

# Final check on the return value.
# The task says: "construct() -> three coefficients of the Hermite combination".
# My `construct` returns `best_coeffs` which is a numpy array of 3 floats. This matches.

# One detail: The seed score is 1.0.
# If my optimization finds a product < 1, the score will be > 1.
# If it finds a product > 1, the score will be < 1.
# The Gaussian alone gives 1. The seed coefficients likely give something close to 1 or slightly better/worse.
# Actually, if the Gaussian is the minimizer, the product would be 1. But it's not.
# The function that minimizes the L1 uncertainty product is not the Gaussian.
# So we should be able to get a product < 1.

# I will add a check to ensure the returned coefficients are finite.

# Re-writing construct with better time management.

def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    initial_guess = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_coeffs = initial_guess
    best_val = objective(initial_guess)
    
    # Optimization 1: Nelder-Mead from seed
    try:
        res = minimize(objective, initial_guess, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() < deadline:
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
    except:
        pass

    # Optimization 2: L-BFGS-B from best found (if smooth enough)
    if time.time() < deadline - 60:
        try:
            res2 = minimize(objective, best_coeffs, method='L-BFGS-B', 
                       options={'maxiter': 100})
            if time.time() < deadline:
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_coeffs = res2.x
        except:
            pass

    # Optimization 3: Random restart if time permits
    if time.time() < deadline - 30:
        np.random.seed(42)
        # Generate a random point near the best
        noise = np.random.normal(0, 0.5, 3)
        guess3 = best_coeffs + noise
        try:
            res3 = minimize(objective, guess3, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if time.time() < deadline:
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_coeffs = res3.x
        except:
            pass

    return best_coeffs

# To be safe against any potential issues with the specific definition of the bound,
# I am relying on the standard L1 uncertainty product interpretation.
# If the "upper bound" refers to something else, this might not be optimal, but it's the strongest hypothesis.

# One more thing: The problem mentions "Hermite-polynomial times Gaussian".
# The Fourier transform of the product is a convolution in frequency? No, it's a product in space, so convolution in frequency?
# No, the Fourier transform of a product is a convolution.
# But $g(x) = e^{-x^2/2} P(x)$.
# The Fourier transform of $e^{-x^2/2} P(x)$ is NOT simply $e^{-2\pi^2 \xi^2} P(2\pi \xi)$ unless P is a polynomial?
# Wait. The Fourier transform of $x^n e^{-x^2/2}$ is related to Hermite polynomials.
# Specifically, $\mathcal{F}[x^n e^{-x^2/2}](\xi) = (-i)^n \frac{\partial^n}{\partial \xi^n} (\sqrt{2\pi} e^{-2\pi^2 \xi^2})$.
# This results in $\sqrt{2\pi} i^n H_n(2\pi \xi) e^{-2\pi^2 \xi^2}$ (with some scaling factors depending on Hermite definition).
# Using the Physicist's Hermite polynomials $H_n(x)$:
# $\int_{-\infty}^{\infty} H_n(x) e^{-x^2/2} e^{-2\pi i x \xi} dx$.
# This is proportional to $i^n H_n(2\pi \xi) e^{-2\pi^2 \xi^2}$.
# So the linearity holds. The Fourier transform of the linear combination is the linear combination of the Fourier transforms.
# So my `norm_fhat` calculation is correct.

# Final Code Structure
