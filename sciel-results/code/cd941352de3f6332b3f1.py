"""Optimized program for uncertainty inequality coefficient search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Find three coefficients (a0, a4, a8) for the Hermite-Gaussian test function
    that minimize the upper bound on A(f)*A(Fourier(f)).
    The degree-12 coefficient is fixed so that g(0) = 0.
    
    Returns:
        np.array of shape (3,) with coefficients [a0, a4, a8]
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 210s limit
    
    # Physics: The test function is g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    # where H_n are physicists' Hermite polynomials.
    # Condition: g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0) = 1
    # H4(0) = -12
    # H8(0) = 1680
    # H12(0) = -498960
    
    # So: a0 - 12*a4 + 1680*a8 - 498960*a12 = 0
    # => a12 = (a0 - 12*a4 + 1680*a8) / 498960
    
    # The quantity to minimize is the upper bound on C4 = A(f)*A(F(f)).
    # For a function g, A(g) = ||g||_1 / ||g||_2 (or similar norm ratio depending on exact definition).
    # Actually, for the uncertainty principle, we typically look at:
    # A(f) = ||f||_1 / ||f||_2^2 * ||f||_2 = ||f||_1 / ||f||_2 ... 
    # Let me think more carefully.
    # 
    # The standard uncertainty product bound involves:
    # A(f) = ||f||_1 / ||f||_2
    # A(ĝ) = ||ĝ||_1 / ||ĝ||_2
    # And by Plancherel, ||ĝ||_2 = (2π)^(-n/2) ||g||_2 in n dimensions (or sqrt(2π) in 1D depending on convention).
    #
    # For 1D Fourier transform with convention ĝ(ξ) = ∫ g(x) e^{-2πixξ} dx:
    # ||ĝ||_2 = ||g||_2 (Plancherel)
    # So A(ĝ) = ||ĝ||_1 / ||g||_2
    #
    # We want to minimize A(g) * A(ĝ) = (||g||_1 / ||g||_2) * (||ĝ||_1 / ||g||_2) = ||g||_1 * ||ĝ||_1 / ||g||_2^2
    #
    # This is scale-invariant: if we scale g by c, both ||g||_1 and ||ĝ||_1 scale by |c|, and ||g||_2^2 scales by c^2, so the ratio is invariant.
    #
    # Let's compute this numerically.
    
    def hermite_polynomial(n, x):
        """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            h0 = np.ones_like(x)
            h1 = 2 * x
            for k in range(2, n + 1):
                h2 = 2 * x * h1 - 2 * (k - 1) * h0
                h0, h1 = h1, h2
            return h1
    
    def compute_bound(a0, a4, a8):
        """Compute the upper bound ||g||_1 * ||ĝ||_1 / ||g||_2^2."""
        # Fix a12 from constraint g(0) = 0
        a12 = (a0 - 12 * a4 + 1680 * a8) / 498960.0
        
        # Evaluate on a grid
        x = np.linspace(-15, 15, 20000)
        dx = x[1] - x[0]
        
        H0 = hermite_polynomial(0, x)
        H4 = hermite_polynomial(4, x)
        H8 = hermite_polynomial(8, x)
        H12 = hermite_polynomial(12, x)
        
        g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2)
        
        # Check g(0) ≈ 0
        # idx0 = np.argmin(np.abs(x))
        # if abs(g[idx0]) > 1e-6:
        #     pass  # constraint should be satisfied by construction
        
        norm1_g = np.sum(np.abs(g)) * dx
        norm2_g = np.sqrt(np.sum(g**2) * dx)
        
        if norm2_g < 1e-10:
            return 1e10
        
        # Compute Fourier transform via FFT or direct integration
        # Use FFT for efficiency
        n_pts = len(x)
        # For FFT, we need uniform spacing and the function sampled appropriately
        # ĝ(ξ) = ∫ g(x) e^{-2πixξ} dx ≈ dx * sum g(x_j) e^{-2πix_j ξ_k}
        # This is essentially an FFT
        # The frequency spacing is Δξ = 1/(N*dx)
        # We need to compute ||ĝ||_1 = ∫ |ĝ(ξ)| dξ ≈ Δξ * sum |ĝ_k|
        
        # Compute FFT
        G = dx * np.fft.fft(g)
        # The FFT gives values at frequencies ξ_k = k/(N*dx) for k=0,...,N/2-1, and negative for the rest
        # Reorder for proper interpretation
        G_shifted = np.fft.fftshift(G)
        freqs = np.fft.fftfreq(n_pts, d=dx)
        freqs_shifted = np.fft.fftshift(freqs)
        
        dxi = freqs_shifted[1] - freqs_shifted[0]
        norm1_G = np.sum(np.abs(G_shifted)) * dxi
        
        bound = norm1_g * norm1_G / (norm2_g**2)
        return bound
    
    # Objective function for optimization
    def objective(params):
        if time.time() > deadline - 10:
            return 1e10
        a0, a4, a8 = params
        try:
            return compute_bound(a0, a4, a8)
        except Exception:
            return 1e10
    
    # Initial best: the seed values
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = objective(seed)
    best_params = seed.copy()
    
    # Strategy 1: Local optimization from multiple starting points
    starting_points = [
        seed,
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-0.5, 0.5, 0.5]),
        np.array([0.5, -0.5, 0.5]),
        np.array([0.5, 0.5, -0.5]),
    ]
    
    for sp in starting_points:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_res = differential_evolution(objective, bounds, maxiter=100, popsize=20,
                                           tol=1e-8, seed=42, polish=True,
                                           init='sobol')
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Final polish with high precision
    if time.time() < deadline - 20:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
