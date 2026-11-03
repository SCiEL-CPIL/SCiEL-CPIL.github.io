import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

# Pre-compute grid for FFT-based integration
# We want to estimate A(g) = int |g(x)| dx and A(F(g)) = int |F(g)(xi)| dxi
# Using FFT:
# g(x) on grid x_i. F(g)(xi) on grid xi_j.
# dx and dxi are related by N and range.

N = 2**15
L = 20.0  # Half-width of domain
x = np.linspace(-L, L, N, endpoint=False)
dx = x[1] - x[0]
# FFT shift: np.fft.fftshift(np.fft.fft(g)) gives F in order from -N/2 to N/2-1
# Frequency grid:
# xi_k = k / (N * dx) ? No, standard FFT freq is k / (N*dx) for k in 0..N-1.
# With fftshift, xi ranges from -1/(2*dx) to 1/(2*dx) - 1/(2*dx)?
# Actually, np.fft.fftfreq(N, d=dx) gives frequencies.
xi = np.fft.fftfreq(N, d=dx)
# Reorder to match fftshift
xi_shifted = np.fft.fftshift(xi)
dxi = xi_shifted[1] - xi_shifted[0]

def _phys_hermite(n, x):
    # Probabilists' Hermite polynomials He_n(x) or Physicists' H_n(x)?
    # The seed uses H_k. Standard Hermite H_n(x) satisfies H_n'(x) = 2n H_{n-1}(x).
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 4x^2 - 2
    # H_3 = 8x^3 - 12x
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = ...
    # We only need n=0, 4, 8, 12.
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
    elif n == 5:
        return 32 * x**5 - 160 * x**3 + 120 * x
    elif n == 6:
        return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    elif n == 7:
        return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    elif n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        # H_12(x) = 2^12 x^12 - ... 
        # Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
        # Let's compute it programmatically to avoid errors
        pass
    # Generic recurrence for safety
    h0 = np.ones_like(x)
    h1 = 2 * x
    if n == 0: return h0
    if n == 1: return h1
    h_n_minus_1 = h1
    h_n_minus_2 = h0
    for k in range(2, n + 1):
        h_n = 2 * x * h_n_minus_1 - 2 * (k - 1) * h_n_minus_2
        h_n_minus_2 = h_n_minus_1
        h_n_minus_1 = h_n
    return h_n_minus_1

# Pre-compute Hermite polynomials on the grid
H0 = _phys_hermite(0, x)
H4 = _phys_hermite(4, x)
H8 = _phys_hermite(8, x)
H12 = _phys_hermite(12, x)

# Values at 0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = _phys_hermite(12, np.array([0.0]))[0]

def _build_g(c):
    c0, c4, c8 = c
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    g = c0 * H0 * np.exp(-x**2 / 4.0) + \
        c4 * H4 * np.exp(-x**2 / 4.0) + \
        c8 * H8 * np.exp(-x**2 / 4.0) + \
        c12 * H12 * np.exp(-x**2 / 4.0)
    return g

def _objective(c):
    g = _build_g(c)
    # A(g)
    A_g = np.sum(np.abs(g)) * dx
    
    # Fourier Transform
    # Use FFT. Note: scipy.fft.fft assumes input is in time domain.
    # The transform is sum g(x) exp(-i 2 pi k x / L_total)
    # Our x is uniform.
    G = np.fft.fft(g)
    G = np.fft.fftshift(G)
    # Scaling: The FFT computes sum. The integral is approx sum * dx.
    # But the frequency variable in FFT is k / (N dx).
    # The continuous FT is int g(x) e^{-i omega x} dx.
    # Discrete approx: sum g(x_j) e^{-i omega_k x_j} dx.
    # This matches G * dx if the frequencies match.
    # The frequencies from np.fft.fftfreq(N, d=dx) are in cycles/unit.
    # omega = 2 * pi * freq.
    # So G_fft corresponds to FT at omega_k = 2 pi * xi_k.
    # The magnitude is |G_fft * dx|.
    A_Fg = np.sum(np.abs(G)) * dx
    # Wait, is there a 2*pi factor?
    # Definition of A(F(f)): Usually F(f)(xi) = int f(x) e^{-2 pi i x xi} dx.
    # If so, the FFT with freq in cycles/unit matches directly.
    # If F(f)(omega) = int f(x) e^{-i x omega} dx, then there is a 1/sqrt(2pi) or similar factor depending on convention.
    # The product A(f)A(F(f)) is invariant under scaling of the transform if we adjust the measure, 
    # but the "upper bound on C4" likely refers to a specific constant in the uncertainty principle.
    # For the standard Fourier transform F(xi) = int f(x) e^{-2 pi i x xi} dx,
    # the uncertainty principle is sigma_x sigma_xi >= 1/(4 pi).
    # The quantity A(f) A(F(f)) is the L1 norm product.
    # Let's assume the standard convention used in such problems.
    # The seed value will calibrate this.
    
    return A_g * A_Fg

def construct():
    start_time = time.time()
    timeout = 170.0 # Safety margin
    
    # Initial guess from seed
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_c = c_seed.copy()
    best_val = _objective(best_c)
    
    # Bounds for DE
    # The coefficients can be scaled, but the shape is what matters.
    # We can bound them to a reasonable range.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Use Differential Evolution
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            popsize=15,
            polish=True,
            update_interval=10
        )
        if result.fun < best_val:
            best_val = result.fun
            best_c = result.x
    except Exception as e:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < timeout:
        # Polish with Nelder-Mead or L-BFGS-B
        try:
            res2 = minimize(_objective, best_c, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x
        except Exception:
            pass

    return best_c

# EVOLVE-BLOCK-END
