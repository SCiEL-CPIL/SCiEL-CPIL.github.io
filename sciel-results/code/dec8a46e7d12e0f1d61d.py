import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    """
    Constructs the coefficients for the Hermite-Gaussian test function
    to minimize the uncertainty constant upper bound.
    """
    # Deadline guard: 210s limit, aim for < 180s
    start_time = time.time()
    max_time = 170.0
    
    # 1. Define the basis functions and constraint logic
    # H_n(x) are probabilists' Hermite polynomials? Or physicists'?
    # The prompt mentions H0, H4, H8, H12.
    # Standard Hermite polynomials $H_n(x)$ (physicists) satisfy $H_n(0)$:
    # H_0(0) = 1
    # H_1(0) = 0
    # H_2(0) = -2
    # H_3(0) = 0
    # H_4(0) = 12
    # H_8(0) = -1344
    # H_12(0) = 10395 * (-1)^6? No, let's compute or look up.
    # Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 4x^2 - 2
    # H_3 = 8x^3 - 12x
    # H_4 = 16x^4 - 48x^2 + 12
    # H_5 = 32x^5 - 160x^3 + 120x
    # H_6 = 64x^6 - 480x^4 + 720x^2 - 120
    # H_7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
    # H_8 = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
    # Wait, H_8(0) = 1680.
    # Let's check H_12(0).
    # H_n(0) for even n: (-1)^{n/2} (n-1)!!
    # H_0(0) = 1
    # H_2(0) = -2
    # H_4(0) = 12
    # H_6(0) = -120
    # H_8(0) = 1680
    # H_10(0) = -30240
    # H_12(0) = 665280
    
    # Let's verify H_4(0) = 12.
    # Formula: H_n(0) = (-1)^{n/2} \frac{n!}{(n/2)!}
    # H_4(0) = (-1)^2 \frac{24}{2} = 12. Correct.
    # H_8(0) = (-1)^4 \frac{40320}{24} = 1680. Correct.
    # H_12(0) = (-1)^6 \frac{479001600}{720} = 665280. Correct.
    
    # The constraint is g(0) = 0.
    # g(x) = (c_0 H_0 + c_4 H_4 + c_8 H_8 + c_12 H_12) e^{-x^2/2}
    # g(0) = c_0(1) + c_4(12) + c_8(1680) + c_12(665280) = 0
    # c_12 = -(c_0 + 12 c_4 + 1680 c_8) / 665280
    
    def get_c12(c0, c4, c8):
        return -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0

    def hermite_poly(x, n):
        # Calculate H_n(x) using recurrence for efficiency and accuracy
        # H_0 = 1, H_1 = 2x
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return 2.0 * x
        
        h_prev = np.ones_like(x)
        h_curr = 2.0 * x
        for k in range(1, n):
            h_next = 2.0 * x * h_curr - 2.0 * k * h_prev
            h_prev = h_curr
            h_curr = h_next
        return h_curr

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Grid for x
        # Choose range and resolution carefully
        N = 4096
        L = 12.0
        x = np.linspace(-L, L, N)
        dx = x[1] - x[0]
        
        # Calculate g(x)
        H0 = hermite_poly(x, 0)
        H4 = hermite_poly(x, 4)
        H8 = hermite_poly(x, 8)
        H12 = hermite_poly(x, 12)
        
        poly_part = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        g = poly_part * np.exp(-0.5 * x**2)
        
        # L1 norm: ||g||_1
        # Using trapezoidal rule
        l1_norm = np.trapz(np.abs(g), x)
        
        if l1_norm < 1e-10:
            return 1e10 # Avoid division by zero if g is effectively 0
        
        # Fourier Transform
        # We need min |g_hat(xi)|
        # g_hat(xi) = FFT(g) * dx (approx)
        # However, FFT assumes periodicity. With large L and decay, this is okay.
        # We need to center the FFT or just take the min magnitude.
        # The Fourier transform of a real even function is real and even.
        # We can compute the FFT, shift it so zero frequency is in the middle, 
        # and take the minimum absolute value.
        
        g_hat = np.fft.fft(g) * dx
        g_hat_shifted = np.fft.fftshift(g_hat)
        
        # The grid for xi
        # xi_k = k / (N * dx) ? 
        # Actually, for fftshift, the frequencies are centered.
        # We just need the minimum magnitude.
        # Note: The DC component (k=0 in original, center in shifted) corresponds to xi=0.
        # The uncertainty product involves the spread. The bound is related to the ratio.
        # We want to minimize ||g||_1^2 / min|g_hat|.
        
        # Exclude the very center if it's zero? 
        # The problem states g(0)=0, not g_hat(0)=0.
        # g_hat(0) = ||g||_1. This is the largest component usually.
        # The minimum will be somewhere else.
        
        min_g_hat = np.min(np.abs(g_hat_shifted))
        
        if min_g_hat < 1e-10:
             return 1e10
        
        return (l1_norm ** 2) / min_g_hat

    # Initial guess from the prompt
    # c0 = -113/100 = -1.13
    # c4 = 1/25 = 0.04
    # c8 = 1/3240 ~ 0.000308
    initial_guess = np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Bounds for optimization
    # The coefficients shouldn't be too large to keep the function well-behaved
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    best_result = None
    best_score = np.inf
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < max_time:
        try:
            # Differential Evolution for global search
            # maxiter and popsize depend on time budget
            # 3 variables, DE is fast
            result = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-8,
                polish=False # We will polish separately to save time in DE
            )
            
            if result.success or result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
            # Check time again
            elapsed = time.time() - start_time
            if elapsed < max_time - 10: # Leave 10s for polish and safety
                # Local refinement using L-BFGS-B
                res_local = minimize(
                    objective, 
                    best_result, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                
                if res_local.success or res_local.fun < best_score:
                    best_score = res_local.fun
                    best_result = res_local.x
                    
        except Exception:
            # If optimization fails, fall back to initial guess
            pass

    if best_result is None:
        return initial_guess
    
    # Ensure the returned array is the correct type and shape
    return np.array(best_result)

# EVOLVE-BLOCK-END
