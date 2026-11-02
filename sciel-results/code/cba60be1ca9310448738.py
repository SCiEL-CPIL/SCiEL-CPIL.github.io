"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import hermite
from scipy.optimize import minimize, differential_evolution
import math

def construct():
    """
    Returns three coefficients (c0, c4, c8) for the Hermite combination.
    The degree-12 weight c12 is fixed so that g(0) = 0.
    The score is the upper bound on C4 = A(f)*A(Fourier(f)).
    Lower is better.
    """
    start_time = time.time()
    deadline = 180.0  # seconds, with safety margin from 210s limit

    # Hermite polynomials: H_n(x)
    # We need H0, H4, H8, H12
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12(x) = 4096x^12 - 73728x^10 + 552960x^8 - 2177280x^6 + 4084032x^4 - 3440640x^2 + 1030320

    def H0(x):
        return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0

    def H4(x):
        if isinstance(x, np.ndarray):
            return 16*x**4 - 48*x**2 + 12
        return 16*x**4 - 48*x**2 + 12

    def H8(x):
        if isinstance(x, np.ndarray):
            return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
        return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680

    def H12(x):
        if isinstance(x, np.ndarray):
            return (4096*x**12 - 73728*x**10 + 552960*x**8 - 
                    2177280*x**6 + 4084032*x**4 - 3440640*x**2 + 1030320)
        return (4096*x**12 - 73728*x**10 + 552960*x**8 - 
                2177280*x**6 + 4084032*x**4 - 3440640*x**2 + 1030320)

    def hermite_combo(c0, c4, c8, c12, x):
        """Compute g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)"""
        if isinstance(x, np.ndarray):
            poly = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
            return poly * np.exp(-x**2/2)
        else:
            poly = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
            return poly * math.exp(-x**2/2)

    def compute_c12(c0, c4, c8):
        """Find c12 such that g(0) = 0.
        g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
        H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 1030320
        So: c0 + 12*c4 + 1680*c8 + 1030320*c12 = 0
        => c12 = -(c0 + 12*c4 + 1680*c8) / 1030320
        """
        return -(c0 + 12*c4 + 1680*c8) / 1030320.0

    def compute_bound(c0, c4, c8):
        """
        Compute the upper bound on A(f)*A(Fourier(f)).
        
        For g(x) = P(x) * exp(-x^2/2), where P is a polynomial,
        the Fourier transform can be computed analytically or numerically.
        
        A(f) = sup_x |f(x)| / ||f||_2  (or similar amplitude measure)
        Actually, in the uncertainty inequality context:
        A(f) is likely the "amplitude" = sup |f(x)|, and A(F) is sup |F(f)(xi)|
        
        The product A(f)*A(F) is bounded by some constant C4.
        
        For a Gaussian times polynomial, we can compute this numerically.
        Let's use a standard definition:
        A(f) = sup_x |f(x)|
        A(F) = sup_xi |hat{f}(xi)|
        
        For f(x) = g(x) normalized appropriately...
        
        Actually, let me think about this more carefully. The task says
        "upper bound on A(f)*A(Fourier(f))" and "score is the upper bound on C4".
        
        The seed returns [-113/100, 1/25, 1/3240] which gives score 1.0.
        Let me compute the bound for the seed to understand the normalization.
        
        For the uncertainty inequality, a common formulation is:
        ||f||_2 * ||x f||_2 >= (1/2) ||f||_2^2  (Heisenberg)
        But here it's about A(f) * A(F) where A is the amplitude (sup norm).
        
        For a function g(x) = P(x) e^{-x^2/2}, the Fourier transform is also
        a polynomial times Gaussian (up to constants).
        
        Let me just compute numerically:
        - A(g) = sup |g(x)|
        - A(F(g)) = sup |F(g)(xi)| where F is the Fourier transform
        
        The product A(g)*A(F(g)) depends on the normalization of F.
        Using the unitary Fourier transform: F(f)(xi) = (1/sqrt(2*pi)) * int f(x) e^{-ixi x} dx
        
        For g(x) = e^{-x^2/2}, F(g) = e^{-xi^2/2}, so A(g)*A(F(g)) = 1*1 = 1.
        
        For the seed coefficients, let me compute this numerically.
        """
        c12 = compute_c12(c0, c4, c8)
        
        # Evaluate on a fine grid
        x = np.linspace(-10, 10, 100000)
        g = hermite_combo(c0, c4, c8, c12, x)
        
        A_g = np.max(np.abs(g))
        
        # Compute Fourier transform numerically
        # Use FFT or analytical
        # For polynomial * Gaussian, the Fourier transform is also polynomial * Gaussian
        # Let's use numerical FFT for accuracy
        dx = x[1] - x[0]
        # Center the function
        G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
        xi = np.fft.fftshift(2*np.pi*np.fft.fftfreq(len(x), d=dx))
        
        A_F = np.max(np.abs(G))
        
        return A_g * A_F

    # Let's first verify the seed
    seed_c0, seed_c4, seed_c8 = -113/100, 1/25, 1/3240
    seed_bound = compute_bound(seed_c0, seed_c4, seed_c8)
    # The score is seed/value, so if value < seed, score > 1
    
    # Now let's optimize
    # We want to minimize the bound A(g) * A(F(g))
    
    def objective(params):
        c0, c4, c8 = params
        try:
            return compute_bound(c0, c4, c8)
        except:
            return 1e10
    
    # Try local optimization from the seed
    best_result = (seed_c0, seed_c4, seed_c8, seed_bound)
    
    # Try several starting points
    start_points = [
        [seed_c0, seed_c4, seed_c8],
        [-1.0, 0.0, 0.0],
        [-0.5, 0.1, 0.0],
        [-1.5, 0.05, 0.0001],
        [-1.2, 0.04, 0.0003],
        [-1.1, 0.03, 0.0002],
        [-1.3, 0.06, 0.0005],
    ]
    
    for sp in start_points:
        if time.time() - start_time > deadline - 30:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success and res.fun < best_result[3]:
                best_result = (res.x[0], res.x[1], res.x[2], res.fun)
        except:
            pass
    
    # Also try BFGS with gradient approximation
    for sp in start_points[:3]:
        if time.time() - start_time > deadline - 20:
            break
        try:
            res = minimize(objective, sp, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < best_result[3]:
                best_result = (res.x[0], res.x[1], res.x[2], res.fun)
        except:
            pass
    
    # Final answer
    c0, c4, c8, _ = best_result
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
