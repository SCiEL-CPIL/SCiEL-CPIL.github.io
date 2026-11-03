import numpy as np
from scipy.optimize import differential_evolution
from numpy.polynomial.laguerre import laggauss

def construct():
    # New Approach: Minimize the L1 norm of the Fourier transform of f(x) = prod(x-xi)^2 * e^-x
    # f(x) is a polynomial P(x) of degree 24 times e^-x.
    # P(x) = sum_{k=0}^{24} c_k x^k
    # hat_f(w) = sum_{k=0}^{24} c_k * k! / (1 + i*w)^(k+1)
    # Score = int_{-inf}^{inf} |hat_f(w)| dw
    
    # We need to compute c_k from the roots.
    # P(x) = prod_{j=1}^{12} (x - x_j)^2 = (prod_{j=1}^{12} (x - x_j))^2
    # Let Q(x) = prod_{j=1}^{12} (x - x_j) = sum_{m=0}^{12} a_m x^m
    # Then P(x) = Q(x)^2 = sum_{k=0}^{24} c_k x^k
    # c_k = sum_{m=0}^k a_m a_{k-m}
    
    # We will optimize the roots x_j to minimize the L1 norm of the Fourier transform.
    
    # Precompute factorials
    max_deg = 24
    factorials = np.array([1.0] * (max_deg + 1))
    for i in range(1, max_deg + 1):
        factorials[i] = factorials[i-1] * i
        
    def compute_coeffs(roots):
        # Compute coefficients a_m of Q(x) = prod (x - x_j)
        # Use numpy poly
        # roots are x_j, so poly is prod (x - x_j)
        # numpy.poly takes roots and returns coefficients in descending order
        # Q(x) = x^12 - e1 x^11 + ...
        # We need a_m for x^m.
        coeffs_desc = np.poly(roots)
        # coeffs_desc[0] is coeff of x^12, coeffs_desc[12] is coeff of x^0
        # a_m = coeffs_desc[12-m]
        a = coeffs_desc[::-1] # a[0] is coeff of x^0, a[12] is coeff of x^12
        
        # Compute c_k = sum_{m=0}^k a_m a_{k-m}
        c = np.convolve(a, a)
        # c has length 24+1 = 25. c[k] is coeff of x^k.
        return c
        
    def fourier_l1_norm(roots):
        c = compute_coeffs(roots)
        
        # Evaluate hat_f(w) on a grid
        # The function decays as 1/w^25.
        # We need a grid that covers the significant part of the spectrum.
        # The "width" of the spectrum is roughly 1 / (spread of roots).
        # Let's use a logarithmic grid or a linear grid with enough points.
        # Since we don't know the scale, let's assume roots are O(10).
        # Spectrum decays quickly.
        
        # Use a fixed grid for consistency and speed
        w = np.linspace(-50, 50, 2000)
        
        # Calculate hat_f(w)
        # term_k = c[k] * k! / (1 + i*w)^(k+1)
        # We can compute this vectorized.
        # shape of w: (2000,)
        # shape of c: (25,)
        
        # To avoid overflow in powers, we can compute (1+1j*w) and then power.
        base = 1.0 + 1j * w
        hat_f = np.zeros_like(w, dtype=complex)
        
        for k in range(max_deg + 1):
            if c[k] == 0: continue
            # (1+1j*w)^(k+1)
            denom = base ** (k + 1)
            hat_f += c[k] * factorials[k] / denom
            
        # L1 norm approximation
        # Trapezoidal rule
        l1_norm = np.trapz(np.abs(hat_f), w)
        
        return l1_norm

    # Initial guess: Laguerre nodes
    x_init, _ = laggauss(12)
    
    # Use Differential Evolution for global optimization
    # Bounds: roots must be positive.
    # Let's bound them between 1e-4 and 100.
    bounds = [(1e-4, 100.0) for _ in range(12)]
    
    # DE can be slow. Let's use a limited number of iterations.
    # maxiter=50, popsize=10 might be too slow for 600s if the function is expensive.
    # The function involves a 2000-point FFT-like sum. 25 * 2000 = 50k ops. Very fast.
    # DE with popsize 15, maxiter 50 -> 750 evaluations.
    # 750 * (convolve + 25*2000 complex ops) is well within 600s.
    
    result = differential_evolution(
        fourier_l1_norm, 
        bounds, 
        seed=42, 
        maxiter=50, 
        popsize=15,
        tol=1e-4,
        polish=False # Polish with L-BFGS-B afterwards? DE returns a good global candidate.
    )
    
    # Polish with L-BFGS-B for local refinement
    x_best = result.x
    
    # Gradient-free polish or just return DE result?
    # DE is usually good enough. Let's try to polish with L-BFGS-B using finite difference gradient.
    # But L-BFGS-B might get stuck. DE is more robust.
    # Let's stick with DE result or do a quick local search.
    
    return list(x_best)
