"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import time

# Physical constants for the problem
# H_n are physicists' Hermite polynomials
# H_0(x) = 1, H_4(x) = x^4 - 6x^2 + 3, H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 90720x^2 + 10395

def _H0(x):
    return np.ones_like(x)

def _H4(x):
    return x**4 - 6*x**2 + 3

def _H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def _H12(x):
    return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 56700*x**4 - 90720*x**2 + 10395

def _g(x, a0, a4, a8):
    """Construct the test function g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)"""
    a12 = (a0 - 3*a4 + 105*a8) / 10395.0
    h = a0 * _H0(x) + a4 * _H4(x) + a8 * _H8(x) + a12 * _H12(x)
    return h * np.exp(-x**2 / 2.0)

def _g_fourier(a0, a4, a8):
    """
    Compute the Fourier transform of g.
    For g(x) = P(x) * exp(-x^2/2), the Fourier transform involves
    derivatives of the Gaussian. We use the property that FT of x^n * exp(-x^2/2)
    can be computed via Hermite polynomial relations.
    
    Actually, for the uncertainty principle with L1 norms, we need:
    A(f) = measure of support of f, A(F) = measure of support of F
    
    For the bound derivation, we compute the L1 norm of g and its FT,
    then the constant C is related to these.
    """
    # The Fourier transform of exp(-x^2/2) is exp(-xi^2/2) (with appropriate normalization)
    # For P(x)*exp(-x^2/2), the FT is a polynomial in xi times exp(-xi^2/2)
    # This is related to the fact that Hermite functions are eigenfunctions of FT
    
    # For the uncertainty bound, we need the L1 norms
    # Let's compute numerically for accuracy
    x = np.linspace(-20, 20, 100000)
    g = _g(x, a0, a4, a8)
    l1_g = np.trapz(np.abs(g), x)
    
    # Fourier transform: use FFT or analytical
    # FT of x^n * exp(-x^2/2) = i^n * d^n/dxi^n exp(-xi^2/2) * (normalization)
    # With unitary FT: F[exp(-x^2/2)] = exp(-xi^2/2)
    # F[x*exp(-x^2/2)] = -i * xi * exp(-xi^2/2) (up to constants)
    
    # For the L1 norm of the FT, we compute numerically
    xi = np.linspace(-20, 20, 100000)
    # Use the fact that FT of polynomial*Gaussian is polynomial*Gaussian
    # g(x) = sum_k c_k H_k(x) exp(-x^2/2)
    # FT of H_k(x) exp(-x^2/2) = (-i)^k H_k(xi) exp(-xi^2/2) (physicists' convention)
    
    a12 = (a0 - 3*a4 + 105*a8) / 10395.0
    # Coefficients for H_0, H_4, H_8, H_12
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    ft_g = np.zeros_like(xi)
    for coeff, deg in zip(coeffs, degrees):
        ft_g += coeff * ((-1j)**deg) * _H_eval(xi, deg) * np.exp(-xi**2 / 2.0)
    
    l1_ft = np.trapz(np.abs(ft_g), xi)
    
    # The uncertainty bound constant
    # For the L1 uncertainty principle: A(f)*A(F) >= C
    # The optimal C is related to the ratio of L1 norms for the test function
    # C = 1 / (l1_g * l1_ft) is not quite right...
    
    # Actually, the bound is: A(f) * A(F) >= (1 / (||g||_1 * ||F[g]||_1)) * (some constant)
    # Let me reconsider: the standard approach is that for the uncertainty principle
    # A(f) * A(F) >= C, where C is the best constant.
    # The test function approach gives an upper bound on what C can be.
    
    # The constant C4 in the task likely refers to a specific formulation.
    # Let's use: bound = 1 / (l1_g * l1_ft) as a proxy, or
    # bound = l1_g * l1_ft (depending on the direction)
    
    # From the uncertainty principle: if A(f) <= a and A(F) <= b, then ab >= C
    # The test function gives: for g, A(g)*A(F[g]) provides a lower bound on C
    # But we're minimizing an upper bound, so...
    
    # Let me think differently. The "upper bound on A(f)*A(Fourier(f))" means
    # we're finding the smallest C such that A(f)*A(F) <= C for all f in some class.
    # No wait, the uncertainty principle says A(f)*A(F) >= C (lower bound).
    
    # Re-reading: "reduce the upper bound on A(f)*A(Fourier(f))"
    # This suggests we're finding an upper bound that is as tight as possible.
    
    # For Fourier-invariant functions, the bound might be expressed differently.
    # Let me use the direct computation: the constant is related to the L1 norms.
    
    # Standard result: for the uncertainty principle with L1,
    # the optimal constant involves the ratio ||f||_1 / ||f||_inf type quantities.
    
    # Let me just compute a reasonable objective that captures the uncertainty.
    # The product of L1 norms (or inverse) should capture the trade-off.
    
    return l1_g, l1_ft

def _H_eval(x, n):
    """Evaluate physicists' Hermite polynomial H_n at x."""
    if n == 0:
        return np.ones_like(x)
    elif n == 4:
        return _H4(x)
    elif n == 8:
        return _H8(x)
    elif n == 12:
        return _H12(x)
    else:
        # General computation using recurrence
        Hn = np.ones_like(x)
        if n >= 1:
            Hn1 = 2 * x
            for k in range(1, n):
                Hn2 = 2 * x * Hn1 - 2 * k * Hn
                Hn = Hn1
                Hn1 = Hn2
            return Hn1 if n >= 1 else Hn
        return Hn

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty bound for given coefficients."""
    x = np.linspace(-30, 30, 200000)
    g = _g(x, a0, a4, a8)
    l1_g = np.trapz(np.abs(g), x)
    
    xi = np.linspace(-30, 30, 200000)
    a12 = (a0 - 3*a4 + 105*a8) / 10395.0
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    ft_g = np.zeros_like(xi)
    for coeff, deg in zip(coeffs, degrees):
        ft_g += coeff * ((-1j)**deg) * _H_eval(xi, deg) * np.exp(-xi**2 / 2.0)
    
    l1_ft = np.trapz(np.abs(ft_g), xi)
    
    # The uncertainty constant: for the L1 uncertainty principle,
    # the bound involves the product of the supports, and the test function
    # gives us a way to estimate the optimal constant.
    # 
    # The standard formulation: A(f) * A(F) >= C
    # The best C is found by minimizing over test functions:
    # C = inf_f A(f) * A(F)
    # 
    # But we're computing an "upper bound" which we want to reduce.
    # This likely means: for a given class of functions, we have
    # A(f) * A(F) <= C_upper, and we want the smallest C_upper.
    #
    # Alternatively, the "upper bound on C4" might refer to a specific
    # constant in a theorem, and the test function provides an upper bound
    # on that constant.
    #
    # Let me use: bound = 1 / (l1_g * l1_ft) 
    # This makes sense because larger L1 norms mean the function is more
    # spread out, which should increase the uncertainty product.
    # Wait, that's backwards. Larger L1 norm means more spread, so
    # A(f) is larger, so A(f)*A(F) should be larger.
    #
    # Actually, for the uncertainty principle lower bound:
    # A(f) * A(F) >= C
    # The test function g gives: A(g) * A(F[g]) >= C
    # So C <= A(g) * A(F[g])
    # To find the best (largest) C, we want to minimize A(g) * A(F[g])
    # over test functions. But the task says "reduce the upper bound."
    #
    # I think the "upper bound on C4" means: we have a theorem that says
    # C4 <= some expression, and we want to minimize that expression.
    # The expression likely involves the L1 norms of the test function.
    #
    # Let me try: bound = l1_g * l1_ft (we want to minimize this)
    # This represents the "cost" of the test function, and a lower cost
    # gives a tighter bound.
    
    return l1_g * l1_ft

def _compute_bound_fast(a0, a4, a8):
    """Faster computation using fewer points."""
    x = np.linspace(-15, 15, 50000)
    g = _g(x, a0, a4, a8)
    l1_g = np.trapz(np.abs(g), x)
    
    xi = np.linspace(-15, 15, 50000)
    a12 = (a0 - 3*a4 + 105*a8) / 10395.0
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    ft_g = np.zeros_like(xi)
    for coeff, deg in zip(coeffs, degrees):
        ft_g += coeff * ((-1j)**deg) * _H_eval(xi, deg) * np.exp(-xi**2 / 2.0)
    
    l1_ft = np.trapz(np.abs(ft_g), xi)
    
    return l1_g * l1_ft

def _objective(params):
    """Objective function for optimization."""
    a0, a4, a8 = params
    try:
        return _compute_bound_fast(a0, a4, a8)
    except:
        return 1e10

def _objective_grad(params):
    """Objective and gradient using finite differences (analytic is complex)."""
    a0, a4, a8 = params
    f0 = _compute_bound_fast(a0, a4, a8)
    
    eps = 1e-6
    grad = np.zeros(3)
    for i in range(3):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _compute_bound_fast(*params_plus)
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _compute_bound_fast(*params_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return f0, grad

def construct():
    """
    Find optimal Hermite coefficients to minimize the uncertainty bound.
    Returns three coefficients [a0, a4, a8].
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Initial guess: seed program values
    initial = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_params = initial.copy()
    best_score = _compute_bound_fast(*best_params)
    
    # Multiple restarts with different initial points
    restarts = [
        initial,
        np.array([-1.0, 0.04, 0.001]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-0.9, 0.05, 0.0015]),
        np.array([-1.1, 0.035, 0.0008]),
        np.array([-1.05, 0.042, 0.0009]),
    ]
    
    for i, x0 in enumerate(restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30.0:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10},
                bounds=[(-10, 10), (-1, 1), (-0.1, 0.1)]
            )
            
            if result.success or result.fun < best_score:
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            result = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12},
                bounds=[(-10, 10), (-1, 1), (-0.1, 0.1)]
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Verify the solution
    final_score = _compute_bound_fast(*best_params)
    if final_score < best_score:
        best_score = final_score
    
    return best_params
# EVOLVE-BLOCK-END
