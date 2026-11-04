"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution (length 2N-1) via FFT."""
    n = len(x)
    size = 2 * n - 1
    # next power of 2 for efficiency
    size_fft = 1
    while size_fft < size:
        size_fft *= 2
    X = np.fft.rfft(x, size_fft)
    conv = np.fft.irfft(X * X, size_fft)[:size]
    return conv

def _score(x):
    """Compute the C3' value: max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_full(x)
    max_abs = np.max(np.abs(conv))
    return max_abs / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    # We want to minimize max|autoconv| / (sum)^2.
    # By scaling invariance of the ratio (numerator scales as s^2, denominator as s^2),
    # we can fix sum(x) = 1 and minimize max|autoconv|.
    # Alternatively, work in a parameterization.
    
    # Strategy: Use spectral parameterization.
    # Let x be represented by its DFT coefficients.
    # If we make the spectrum flat (white noise), the autoconvolution is a delta-like function
    # (or rather, the inverse FFT of a flat spectrum is white noise, whose autoconvolution 
    # is a triangle-like peak). Wait, flat spectrum -> x is white noise.
    # Autoconvolution of white noise is a triangular pulse peaking at lag 0.
    # max|autoconv| for white noise with sum 0? No, sum is not 0.
    
    # Let's think about the objective: min max_k |sum_i x_i x_{i+k}| / (sum x)^2.
    # If x is a "bang-bang" signal (±1), sum is small, denominator is small.
    # If x is all ones, sum=N, autoconv is triangular with peak N. Score = N/N^2 = 1/N.
    # For N=400, score is 0.0025.
    
    # Can we do better?
    # If we use x_i = (-1)^i, sum is 0 (undefined score, or large if perturbed).
    # If we use x_i = 1 for i<200, -1 for i>=200. Sum = 0.
    # We need sum != 0.
    
    # Let's try to minimize max|autoconv| subject to sum(x) = 1.
    # Then score = max|autoconv|.
    
    # Parameterization: x = 1 + y, where sum(y) = 0.
    # Or use Fourier series: x_k = a_0 + sum_{j=1}^{M} (a_j cos(2*pi*j*k/N) + b_j sin(2*pi*j*k/N)).
    # This ensures x is real.
    # If we restrict to low frequencies, x is smooth.
    # If we use high frequencies, x is oscillatory.
    
    # Let's use differential evolution on the Fourier coefficients.
    # M = 10 harmonics -> 21 parameters (a0, a1..a10, b1..b10).
    # Constraint: sum(x) = 1.
    # sum(x) = N * a0 (since sum of cos/sin over full period is 0).
    # So a0 = 1/N.
    # Remaining parameters: a1..aM, b1..bM.
    
    M = 15
    n_params = 2 * M
    
    def make_x(params):
        # params are a1..aM, b1..bM
        x = np.zeros(N)
        k = np.arange(N)
        x += 1.0 / N  # a0
        for j in range(1, M + 1):
            a_j = params[j - 1]
            b_j = params[M + j - 1]
            x += a_j * np.cos(2 * np.pi * j * k / N) + b_j * np.sin(2 * np.pi * j * k / N)
        return x
    
    def objective(params):
        x = make_x(params)
        # Ensure sum is close to 1 (it is by construction, but numerical errors?)
        # s = np.sum(x)
        # if abs(s) < 1e-6: return 1e10
        # conv = _autoconv_full(x)
        # return np.max(np.abs(conv)) / (s*s)
        
        # Faster: use the fact that sum(x)=1 exactly in floating point if we construct carefully?
        # Actually, sum of cos/sin might not be exactly 0.
        # Let's just compute the score directly.
        return _score(x)
    
    # Bounds for Fourier coefficients.
    # x should be bounded to avoid huge values.
    # Let's say |x_i| < 10.
    # a0 = 1/N = 0.0025.
    # If we allow a_j, b_j in [-5, 5], x can be large.
    # Let's start with smaller bounds.
    bounds = [(-1, 1)] * n_params
    
    # Differential evolution
    best_x = np.ones(N)  # initial guess: all ones
    best_score = _score(best_x)
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            popsize=15,
            updating='immediate',
            polish=False
        )
        
        if result.success or result.fun < best_score:
            best_params = result.x
            best_x = make_x(best_params)
            best_score = _score(best_x)
        
        # Check time
        if time.time() < deadline - 20:
            # Polish with L-BFGS-B or similar? 
            # The objective is non-smooth (max). 
            # We can use a smooth approximation or just accept the DE result.
            # Let's try a local polish using gradient descent on a smooth surrogate?
            # Or just run a few more DE iterations?
            
            # Let's try to refine with a finer search around the best point.
            # Or use scipy.optimize.minimize with method='Powell' which is derivative-free.
            res2 = minimize(
                objective, 
                best_params, 
                method='Powell', 
                options={'maxiter': 200, 'tol': 1e-8}
            )
            if res2.fun < best_score:
                best_params = res2.x
                best_x = make_x(best_params)
                best_score = _score(best_x)
                
    except Exception as e:
        pass
        
    # Final check: ensure we return a valid sequence of signed step heights.
    # The problem asks for "sequence of signed step heights".
    # np.ones(N) is a sequence of heights.
    # My constructed x is a sequence of real numbers.
    # Is "signed step heights" implying discrete values like +/-1?
    # "Signed equal-width step functions" usually means a function that is constant on intervals.
    # The "heights" are the values. They can be any real numbers.
    # "Signed" just means they can be positive or negative.
    
    # One more check: if best_x has very small sum, the score calculation might be unstable.
    # But we fixed sum to 1 via a0.
    
    return best_x

# EVOLVE-BLOCK-END
