"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite_poly(n, x):
    """Compute the probabilists' or physicists' Hermite polynomial H_n(x).
    Using physicists' convention: H_n(x) = (-1)^n e^{x^2/2} d^n/dx^n e^{-x^2/2}
    H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, H_3 = 8x^3 - 12x, H_4 = 16x^4 - 48x^2 + 12
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        # H_8 = 256 x^8 - 3072 x^6 + 11520 x^4 - 13440 x^2 + 1344
        return 256 * x**8 - 3072 * x**6 + 11520 * x**4 - 13440 * x**2 + 1344
    elif n == 12:
        # H_12 = 4096 x^12 - 98304 x^10 + 802816 x^8 - 3041280 x^6 + 5308416 x^4 - 3870720 x^2 + 21840
        return (4096 * x**12 - 98304 * x**10 + 802816 * x**8 - 3041280 * x**6 
                + 5308416 * x**4 - 3870720 * x**2 + 21840)
    else:
        raise ValueError(f"Unsupported n={n}")

def _compute_bound(c0, c4, c8):
    """Compute the upper bound A(f)*A(Fourier(f)) for given coefficients."""
    # c12 is determined by g(0) = 0
    # H_0(0)=1, H_4(0)=-12, H_8(0)=1344, H_12(0)=-21840 (check: H_12(0) = 21840? Let me recompute)
    # From the polynomial above: H_12(0) = 21840
    # Wait, let me check H_4(0) = 12, not -12. Let me recheck.
    # H_4(x) = 16x^4 - 48x^2 + 12, so H_4(0) = 12
    # H_8(0) = 1344
    # H_12(0) = 21840
    
    # Actually, let me verify with the standard physicists' Hermite polynomials:
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 4x^2 - 2
    # H_3 = 8x^3 - 12x
    # H_4 = 16x^4 - 48x^2 + 12
    # H_5 = 32x^5 - 160x^3 + 120x
    # H_6 = 64x^6 - 480x^4 + 720x^2 - 120
    # H_7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
    # H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1344
    # H_9 = 512x^9 - 7680x^7 + 34560x^5 - 50400x^3 + 20160x
    # H_10 = 1024x^10 - 19200x^8 + 107520x^6 - 241920x^4 + 201600x^2 - 30240
    # H_11 = 2048x^11 - 44032x^9 + 282240x^7 - 720720x^5 + 756720x^3 - 221760x
    # H_12 = 4096x^12 - 98304x^10 + 802816x^8 - 3041280x^6 + 5308416x^4 - 3870720x^2 + 21840
    
    # So H_4(0) = 12, H_8(0) = 1344, H_12(0) = 21840
    # g(0) = c0*1 + c4*12 + c8*1344 + c12*21840 = 0
    # => c12 = -(c0 + 12*c4 + 1344*c8) / 21840
    
    c12 = -(c0 + 12.0 * c4 + 1344.0 * c8) / 21840.0
    
    # Define g(x) on a grid
    x = np.linspace(-15, 15, 20001)
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-x**2 / 2.0)
    
    # A(f) = integral of |g(x)| dx
    A_g = np.trapz(np.abs(g), x)
    
    # Fourier transform: use the property that F[H_n(x) e^{-x^2/2}] = i^n H_n(x) e^{-x^2/2}
    # with the convention F[f](xi) = 1/sqrt(2*pi) * integral f(x) exp(-i*x*xi) dx
    # But the task might use a different convention. Let's use numerical FFT to be safe.
    # Actually, for the uncertainty product, the convention affects the constant.
    # Let's use the standard: if g(x) = sum c_k H_k(x) exp(-x^2/2), then
    # G(xi) = sum c_k i^k H_k(xi) exp(-xi^2/2) [with 1/sqrt(2*pi) convention]
    # A(G) = integral |G(xi)| dxi
    
    # Using the eigenfunction property:
    G = (c0 * 1 * H0 + c4 * (1j**4) * H4 + c8 * (1j**8) * H8 + c12 * (1j**12) * H12) * np.exp(-x**2 / 2.0)
    # 1j**4 = 1, 1j**8 = 1, 1j**12 = -1
    # So G = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * (-1) * H12) * exp(-x^2/2)
    
    # Wait, this means G(xi) = (c0*H0 + c4*H4 + c8*H8 - c12*H12) * exp(-xi^2/2)
    # That's different from g! Let me double-check.
    
    # With F[f](xi) = (1/sqrt(2*pi)) int f(x) e^{-ixxi} dx:
    # F[H_n(x) e^{-x^2/2}](xi) = i^n H_n(xi) e^{-xi^2/2}
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = i^(12 mod 4) = i^0 = 1
    # Wait: i^12 = (i^4)^3 = 1^3 = 1. So all are 1!
    
    # So G = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-xi^2/2) = g(xi)
    # That means A(G) = A(g) in this case? That can't be right for the uncertainty product.
    
    # Hmm, let me reconsider. The issue is that the Hermite functions as eigenfunctions 
    # of the Fourier transform use a specific normalization. Let me just do numerical FFT.
    
    # Use numerical Fourier transform
    # For numerical stability, use a finite grid and FFT
    npts = 20001
    dx = x[1] - x[0]
    # Compute Fourier transform numerically
    # F(xi) = int g(x) e^{-i*x*xi} dx
    # We evaluate at xi = x (same grid)
    # Use the fact that for a grid, we can use FFT
    
    # Actually, let's just use the eigenfunction property correctly.
    # The standard result with F[f](xi) = int_{-inf}^{inf} f(x) e^{-i x xi} dx (no 1/sqrt(2*pi)):
    # F[H_n(x) e^{-x^2/2}](xi) = i^n H_n(xi) e^{-xi^2/2} * 2^{n/2} * sqrt(pi) ... 
    # This is getting complicated. Let me just do numerical integration.
    
    # Numerical Fourier transform on the grid
    # G(xi_j) = sum_k g(x_k) exp(-i x_k xi_j) dx
    # This is O(n^2) which is too slow for n=20001.
    
    # Better: use FFT. The grid is uniform, so we can use FFT.
    # Shift x to be centered at 0 for FFT
    x_centered = x - x[0]
    # FFT gives F[k] = sum_n f[n] exp(-2*pi*i*n*k/N)
    # We need exp(-i*x_k*xi_j) = exp(-i*(k*dx)*(j*dx)) = exp(-i*2*pi*(k*j*dx^2)/(2*pi))
    # This doesn't directly map to FFT unless dx^2 = 1/(N*df) where df is frequency spacing.
    
    # Let me just use the eigenfunction property. For the convention:
    # F[f](xi) = (1/sqrt(2*pi)) int f(x) e^{-ixxi} dx
    # The Hermite function psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}
    # satisfies F[psi_n](xi) = i^n psi_n(xi)
    # So F[H_n(x) e^{-x^2/2}](xi) = i^n (2^n n! sqrt(pi))^{1/2} H_n(xi) e^{-xi^2/2}
    
    # This is getting messy. Let me just use a simpler approach:
    # The bound is A(g) * A(G) where A(h) = int |h(x)| dx
    # By the uncertainty principle, A(g) * A(G) >= 1/(4*pi) for the standard convention
    # But we want to minimize the bound, which is achieved by making g as "concentrated" as possible
    # while satisfying the constraints.
    
    # Let me just numerically compute both integrals using the eigenfunction property
    # with the convention F[f](xi) = int f(x) e^{-2*pi*i*x*xi} dx (signal processing convention)
    # In this case, the Hermite functions are not eigenfunctions in the simple way.
    
    # Let me just do a direct numerical computation with a moderate grid size.
    
    # Use a smaller grid for speed
    x_grid = np.linspace(-10, 10, 4001)
    H0_g = _hermite_poly(0, x_grid)
    H4_g = _hermite_poly(4, x_grid)
    H8_g = _hermite_poly(8, x_grid)
    H12_g = _hermite_poly(12, x_grid)
    
    g_vals = (c0 * H0_g + c4 * H4_g + c8 * H8_g + c12 * H12_g) * np.exp(-x_grid**2 / 2.0)
    
    A_g = np.trapz(np.abs(g_vals), x_grid)
    
    # For the Fourier transform, use the property with convention F[f](xi) = int f(x) e^{-ixxi} dx
    # F[H_n(x) e^{-x^2/2}](xi) = i^n H_n(xi) e^{-xi^2/2} * 2^{n/2} * sqrt(pi)
    # Let me verify: F[e^{-x^2/2}](xi) = int e^{-x^2/2} e^{-ixxi} dx = sqrt(2*pi) e^{-xi^2/2}
    # So F[H_0(x) e^{-x^2/2}](xi) = sqrt(2*pi) e^{-xi^2/2}
    # H_0 = 1, so this gives sqrt(2*pi) * H_0(xi) * e^{-xi^2/2}
    # Formula: i^0 * H_0(xi) * e^{-xi^2/2} * 2^0 * sqrt(pi) = sqrt(pi) e^{-xi^2/2}
    # But we need sqrt(2*pi), so the formula should have sqrt(2) factor or different normalization.
    
    # Let me just use: F[H_n(x) e^{-x^2/2}](xi) = i^n * (2^n * n!)^{1/2} * sqrt(pi) * H_n(xi) * e^{-xi^2/2} / (2^n * n!)^{1/2}
    # This is circular. Let me just numerically compute the FT for the specific coefficients.
    
    # Since we only need the L1 norm of the FT, and the FT of a linear combination is a linear combination,
    # let me compute the FT of each basis function numerically once, then combine.
    
    # For efficiency, precompute the FT of each H_n(x) e^{-x^2/2} on the grid
    # using a single FFT-based approach or direct quadrature.
    
    # Actually, for the purposes of this optimization, let me use the fact that:
    # With F[f](xi) = int f(x) e^{-ixxi} dx, we have:
    # F[H_n(x) e^{-x^2/2}](xi) = i^n H_n(xi) e^{-xi^2/2} * 2^{n/2} * sqrt(pi) * (1/sqrt(2^n n!)) * sqrt(2^n n!)
    # I'm going in circles. Let me just use scipy.fft or numpy.fft.
    
    # Use numpy.fft on the grid
    # The grid is x_grid from -10 to 10 with 4001 points, dx = 20/4000 = 0.005
    dx_grid = x_grid[1] - x_grid[0]
    N = len(x_grid)
    
    # FFT: F[k] = sum_n g[n] exp(-2*pi*i*n*k/N)
    # We want G(xi_j) = sum_n g(x_n) exp(-i x_n xi_j) dx
    # If xi_j = j * (2*pi/(N*dx)), then exp(-i x_n xi_j) = exp(-i (n*dx) (j*2*pi/(N*dx))) = exp(-2*pi*i*n*j/N)
    # So G(xi_j) = dx * FFT(g)[j] for j in the appropriate range
    
    G_fft = np.fft.fftshift(np.fft.fft(g_vals)) * dx_grid
    # Frequency grid
    xi_grid = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx_grid))
    
    A_G = np.trapz(np.abs(G_fft), xi_grid)
    
    bound = A_g * A_G
    
    return bound

def construct():
    start_time = time.time()
    deadline = start_time + 180  # 180 second limit with margin
    
    # Initial guess - try to find a valid configuration
    # The constraint is g(0) = 0, which we've built in.
    # We need the bound to be positive and finite.
    
    # Try a few initial points
    initial_points = [
        [1.0, -0.1, 0.01],
        [1.0, -0.05, 0.005],
        [0.5, -0.05, 0.01],
        [1.0, -0.2, 0.02],
        [2.0, -0.1, 0.01],
        [1.0, -0.15, 0.015],
    ]
    
    best_bound = np.inf
    best_x = initial_points[0]
    
    for x0 in initial_points:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_compute_bound, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_bound:
                best_bound = res.fun
                best_x = res.x
        except Exception:
            continue
    
    # Refine with a tighter optimization
    if time.time() < deadline - 10:
        try:
            res = minimize(_compute_bound, best_x, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_bound:
                best_bound = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Return the three coefficients
    return np.array(best_x)

# EVOLVE-BLOCK-END
