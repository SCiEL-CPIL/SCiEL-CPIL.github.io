"""Rewritten program: numerically optimize the Hermite-Gaussian coefficients
to reduce the uncertainty (Fourier concentration) upper bound C4.

New approach family: instead of returning fixed analytic constants, we run a
numerical local optimization (L-BFGS-B) over the three free coefficients,
starting from the Goncalves et al. (2017) state-of-the-art point.  The
objective reconstructs the test function g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)
exp(-x^2/2) with a12 fixed by g(0)=0, then measures the product of the L1 mass
of g and of its Fourier transform, a standard uncertainty concentration
quantity that the C4 bound is built from.  Lower product => tighter bound.
"""
# EVOLVE-BLOCK-START
import numpy as np


def _phys_hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _build_g(coeffs, x):
    """Return g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) * exp(-x^2/2).

    a12 is chosen so that g(0) = 0.
    """
    a0, a4, a8 = coeffs
    H0 = _phys_hermite(0, x)
    H4 = _phys_hermite(4, x)
    H8 = _phys_hermite(8, x)
    H12 = _phys_hermite(12, x)
    H0_0, H4_0, H8_0, H12_0 = 1.0, 3.0, 105.0, 10395.0
    # g(0) = (a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0) * 1 = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    return poly * np.exp(-0.5 * x * x)


def _objective(coeffs):
    """Uncertainty concentration proxy: A(g) * A(Fourier(g))."""
    # Use a sufficiently large domain and fine grid for accurate FFT approximation
    L = 15.0
    N = 16384
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    
    g = _build_g(coeffs, x)
    
    # L1 mass of g
    A_g = np.trapz(np.abs(g), x)
    
    # Compute Fourier Transform using FFT with zero-padding
    N_fft = N * 4
    g_padded = np.zeros(N_fft)
    g_padded[:N] = g
    
    # FFT
    g_fft = np.fft.fft(g_padded)
    
    # Approximate L1 norm of the continuous Fourier transform
    # As derived: A_ghat ~= (1/N_fft) * sum(|FFT(g)|)
    # Note: This assumes the standard FFT scaling.
    # Let's double check.
    # Continuous FT: G(xi) = int g(x) e^{-2pi i x xi} dx
    # Discrete: G_k = dx * sum_n g(x_n) e^{-2pi i x_n xi_k}
    # xi_k = k / (N_fft * dx)
    # G_k = dx * FFT(g)_k (ignoring phase shift from centering)
    # L1 norm = sum |G_k| dxi = sum |dx * FFT(g)_k| * (1/(N_fft * dx))
    #        = (1/N_fft) * sum |FFT(g)_k|
    
    A_ghat = np.sum(np.abs(g_fft)) / N_fft
    
    return A_g * A_ghat


def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    try:
        from scipy.optimize import minimize
        base = _objective(seed)
        best = seed
        best_val = base
        
        # Use more restarts with a slightly wider spread to explore the landscape
        # The FFT-based objective is fast, so we can afford more evaluations.
        rng = np.random.default_rng(42)
        
        # Try a few different initial points
        # 1. Seed
        # 2. Seed + small noise
        # 3. Seed + larger noise
        # 4. Random points in a reasonable box
        
        init_points = [seed]
        for _ in range(3):
            init_points.append(seed + rng.normal(0.0, 0.1, size=3))
        for _ in range(3):
            init_points.append(seed + rng.uniform(-0.5, 0.5, size=3))
        
        for x0 in init_points:
            # Clip values to avoid extreme numbers that might cause overflow in Hermite polys
            # H12 can be large. Let's keep coeffs within reasonable bounds.
            x0_clipped = np.clip(x0, -10.0, 10.0)
            res = minimize(
                _objective,
                x0_clipped,
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.fun < best_val:
                cand = np.asarray(res.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _objective(cand)
                    if np.isfinite(v) and v < best_val:
                        best_val = v
                        best = cand
                        
        return best
    except Exception:
        # Fall back to the known-good state of the art if anything fails.
        return seed
# EVOLVE-BLOCK-END
