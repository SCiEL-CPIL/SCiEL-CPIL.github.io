"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

def construct():
    start_time = time.time()
    deadline = start_time + 150  # Safety margin
    
    # Define Hermite polynomials (Physicists')
    # H_0 = 1
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H_12 = 4096x^12 - 98304x^10 + 806400x^8 - 3182400x^6 + 5644800x^4 - 4032000x^2 + 103950
    
    # Coefficients for H_n(0)
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 103950
    
    def g(x, c0, c4, c8):
        # Calculate c12 to ensure g(0) = 0
        # c0*1 + c4*12 + c8*1680 + c12*103950 = 0
        c12 = -(c0 * 1.0 + c4 * 12.0 + c8 * 1680.0) / 103950.0
        
        # Polynomial P(x)
        # We can compute P(x) directly using the recurrence or explicit forms.
        # Explicit forms are faster for fixed degrees.
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        x10 = x8 * x2
        x12 = x8 * x4
        
        H0 = np.ones_like(x)
        H4 = 16.0 * x4 - 48.0 * x2 + 12.0
        H8 = 256.0 * x8 - 3072.0 * x6 + 11520.0 * x4 - 13440.0 * x2 + 1680.0
        H12 = 4096.0 * x12 - 98304.0 * x10 + 806400.0 * x8 - 3182400.0 * x6 + 5644800.0 * x4 - 4032000.0 * x2 + 103950.0
        
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        return P * np.exp(-0.5 * x2)

    def objective(coeffs):
        c0, c4, c8 = coeffs
        
        # Check deadline
        if time.time() > deadline:
            return 1e10
            
        # Sample grid
        # Need a range large enough to capture the tails
        # Hermite-Gaussians decay as exp(-x^2/2). 10 standard deviations is plenty.
        x = np.linspace(-10, 10, 10001)
        dx = x[1] - x[0]
        
        y = g(x, c0, c4, c8)
        
        # L2 norm squared
        norm_sq = np.sum(y**2) * dx
        if norm_sq < 1e-10:
            return 1e10
        norm = np.sqrt(norm_sq)
        
        # Compute Fourier Transform
        # Use FFT for speed? Or numerical integration?
        # FFT is faster.
        # Note: FFT assumes uniform spacing.
        # We need the FT of y(x).
        # Using scipy.fft or numpy.fft
        Y = np.fft.fft(y)
        # Shift zero frequency to center
        Y = np.fft.fftshift(Y)
        # Frequency axis
        # dx = 0.002
        # df = 1 / (N * dx)
        df = 1.0 / (len(x) * dx)
        # The FT definition in numpy is sum y_n exp(-2pi i n k / N) * dx
        # So Y_fft is approx integral.
        # Parseval: integral |y|^2 = integral |Y|^2
        # Let's check the scaling.
        # np.fft.fft(y) * dx approximates the FT.
        Y = Y * dx
        
        # L2 norm of FT
        norm_ft_sq = np.sum(Y**2) * df  # Wait, df is the step in frequency.
        # Actually, if Y is the FT, then |Y|^2 integrated over freq should equal |y|^2 integrated over time.
        # Let's just compute the moment of x and freq directly to be safe.
        
        # Uncertainty Product: Delta_x * Delta_freq
        # Delta_x^2 = <x^2> - <x>^2
        # Delta_freq^2 = <freq^2> - <freq>^2
        
        # Since g(0)=0 and symmetry might be broken, <x> might not be 0.
        # But H_n are even for even n. So g(x) is even. <x> = 0.
        # <x^2> = sum x^2 y^2 / sum y^2
        
        x2_y2 = x**2 * y**2
        ex2 = np.sum(x2_y2) * dx / norm_sq
        
        # For frequency, we can use the property that <freq^2> in FT domain is related to <x^2> in time domain?
        # No, <freq^2> = (1/4pi^2) * <(d/dx y)^2> ?
        # Let's compute the derivative numerically or analytically.
        # Analytical derivative is better.
        # d/dx (P(x) e^{-x^2/2}) = (P'(x) - x P(x)) e^{-x^2/2}
        
        # P'(x) = c0*0 + c4*(64x^3 - 96x) + c8*(2048x^7 - 18432x^5 + 46080x^3 - 26880x) + c12*(49152x^11 - 983040x^9 + 6451200x^7 - 19094400x^5 + 22579200x^3 - 8064000x)
        
        # This is getting complex. Let's use the FT property:
        # If y(x) is the function, and Y(k) is FT, then <k^2> = (1/4pi^2) * (norm of derivative squared) / norm_sq ?
        # Actually, \int k^2 |Y(k)|^2 dk = (1/4\pi^2) \int |y'(x)|^2 dx.
        
        # Let's compute y'(x) numerically using gradient.
        dy_dx = np.gradient(y, dx)
        norm_dy_sq = np.sum(dy_dx**2) * dx
        
        # Factor 1/(4 pi^2) depends on FT definition.
        # If FT is \int y e^{-ikx} dx, then \int k^2 |Y|^2 dk = \int |y'|^2 dx.
        # If FT is \int y e^{-2pi i k x} dx, then \int k^2 |Y|^2 dk = (4 pi^2) \int |y'|^2 dx.
        # Numpy FFT uses e^{-2pi i ...}.
        # So <k^2> = (4 pi^2) * norm_dy_sq / norm_sq.
        
        ek2 = (4.0 * np.pi**2) * norm_dy_sq / norm_sq
        
        delta_x = np.sqrt(ex2)
        delta_k = np.sqrt(ek2)
        
        return delta_x * delta_k

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Run optimization
    best_result = None
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.success or result.fun < 1e10:
            best_result = result.x
    except Exception:
        pass
        
    if best_result is None:
        best_result = x0
        
    # Ensure we are within time limit
    if time.time() > deadline - 10:
        # Return best known or seed
        return np.array(best_result)
        
    return np.array(best_result)

# EVOLVE-BLOCK-END
