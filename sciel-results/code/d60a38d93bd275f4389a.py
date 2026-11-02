"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def _autoconvolution_bound(signs):
    """
    Compute the upper bound on C3 given a signed step function.
    The step function is defined by 'signs' (array of +1/-1) over N intervals on [-1/4, 1/4].
    The autoconvolution upper bound is related to the maximum of the autocorrelation.
    
    For a step function f(x) = sum_i signs[i] * 1_{x in I_i}, the autoconvolution is:
    (f * f)(t) = ∫ f(x) f(x-t) dx
    
    The bound on C3 is typically related to the L2 norm or the maximum of the autoconvolution.
    Based on the task description "improve an autoconvolution upper bound", we minimize the
    maximum value of the absolute autoconvolution, or a related spectral quantity.
    
    Let's use the Parseval identity: ||f*f||_2^2 = ||F|^2 * |F|^2 ||_2 where F is the Fourier transform.
    Actually, the L2 norm of the autoconvolution is the square of the L2 norm of the function.
    ||f*f||_2 = ||f||_2^2.
    Since f is a step function with values +/-1 on intervals of equal length, ||f||_2^2 is constant.
    
    The "upper bound on C3" likely refers to the supremum of the autoconvolution, i.e., max_t |f*f|(t).
    We want to minimize this maximum.
    
    The autoconvolution of a step function can be computed efficiently using FFT.
    """
    # The function is 1 on positive intervals, -1 on negative intervals.
    # The length of each interval is 1/2 / N = 1/(2N).
    # Let's work in the discrete domain for the FFT.
    # f is a sequence of N values, each of width dx.
    # The continuous autoconvolution at lag k (where k is an integer number of intervals)
    # is proportional to the sum of signs[i] * signs[j] where |i-j| corresponds to the lag.
    
    # More precisely, let's compute the autocorrelation of the sign vector.
    # The maximum of the continuous autoconvolution will occur at one of the interval boundaries
    # or centers. For a step function, the autoconvolution is a piecewise linear function.
    # The maximum of |f*f| is achieved at t=0, where (f*f)(0) = ∫ f(x)^2 dx = 1/2.
    # Wait, if f(x) is +/-1, then f(x)^2 = 1, so (f*f)(0) = ∫ 1 dx = 1/2. This is constant.
    
    # The task says "improve an autoconvolution upper bound". This suggests the bound is not just the value at 0.
    # Perhaps the bound involves the integral of |f*f| or the L1 norm?
    # Or maybe the "upper bound on C3" is a specific quantity from the paper.
    
    # Let's assume the score is the maximum of the absolute value of the autoconvolution
    # excluding the peak at t=0, or perhaps the L2 norm of the autoconvolution away from 0.
    
    # Let's try minimizing the L2 norm of the autoconvolution.
    # ||f*f||_2 = ||f||_2^2 = 1/2. This is also constant.
    
    # Let's reconsider. The "third autocorrelation inequality" might refer to a bound involving
    # the autocorrelation at a specific lag, or the sum of autocorrelations.
    
    # A common quantity in such problems is the "energy" of the autocorrelation, i.e., the sum of squares.
    # By Parseval, this is related to the 4th power of the Fourier coefficients.
    
    # Let's use the following metric: the sum of the squares of the autocorrelation coefficients.
    # This is equivalent to the integral of (f*f)^2 dx.
    # ∫ (f*f)^2 dx = ∫∫∫∫ f(x)f(x-t)f(y)f(y-t) dxdydt
    # This is the 4th moment of the function.
    
    # Let's compute this using FFT.
    # The autocorrelation R(t) = ∫ f(x)f(x-t) dx.
    # The integral of R(t)^2 dt = ∫∫ R(t)R(s) dt ds = ||R||_2^2.
    # By Parseval, ||R||_2^2 = (1/2π) ∫ |F(ω)|^4 dω.
    
    # Let's compute the FFT of the sign vector and sum the 4th powers of the magnitudes.
    # This is a good proxy for the "energy" of the autocorrelation.
    
    signs_float = signs.astype(np.float64)
    # Pad to next power of 2 for efficient FFT? N=400, next power is 512.
    # Actually, for autocorrelation, we can use FFT of length 2N.
    n_fft = 2 * N
    F = np.fft.rfft(signs_float, n=n_fft)
    # The integral of (f*f)^2 is proportional to sum(|F|^4).
    score = np.sum(np.abs(F)**4)
    return score

def _construct_from_sine(freq, phase):
    """Construct a sign vector from a sine wave."""
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    wave = np.sin(2 * np.pi * freq * x + phase)
    signs = np.sign(wave)
    # Ensure we don't have zeros (though unlikely for sine)
    signs[signs == 0] = 1
    return signs

def construct():
    """
    Construct a signed step function to minimize the autoconvolution upper bound.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_signs = np.ones(N)
    best_score = _autoconvolution_bound(best_signs)
    
    # Try a set of low-frequency sine waves
    # The frequency should be an integer to ensure the function is periodic on the domain?
    # Not necessarily, but low frequencies will have smooth transitions.
    # Let's try frequencies from 1 to 20.
    
    for freq in range(1, 21):
        if time.time() > deadline:
            break
        for phase in np.linspace(0, np.pi, 10):
            if time.time() > deadline:
                break
            signs = _construct_from_sine(freq, phase)
            score = _autoconvolution_bound(signs)
            if score < best_score:
                best_score = score
                best_signs = signs.copy()
    
    # Local search: flip individual signs to see if it improves the score
    # This is a greedy hill-climbing approach.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Flip sign i
            old_sign = best_signs[i]
            best_signs[i] = -old_sign
            score = _autoconvolution_bound(best_signs)
            if score < best_score:
                best_score = score
                improved = True
            else:
                best_signs[i] = old_sign
    
    return best_signs

# EVOLVE-BLOCK-END
